import math
import os
import time

import pandas as pd

from flask import Flask, render_template, request, session, redirect, url_for, flash, abort, jsonify
from database import get_db
from werkzeug.security import generate_password_hash, check_password_hash
from recommender import (
    movies_df,
    stats_for,
    recommend_movies,
    recommend_for_user,
    recommend_cold_start,
    rating_distribution,
    recommend_by_mood,
    recommend_surprise,
    recommend_social,
    recommend_by_chat,
    taste_dna,
    taste_dna_timeline,
    MOODS,
    DNA_DIMS,
)
from tmdb import enrich_movies, has_api_key, cached_poster
from poster import get_poster, split_title_year, primary_genre, GENRE_LABELS
from config import SECRET_KEY
import chat_engine


app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7

# 电影库预计算评分统计，避免每次请求对 10 万条评分重复 groupby
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "data", "ratings.csv"))
_rating_stats = (
    ratings_df.groupby("movieId")["rating"]
    .agg(avg_rating="mean", rating_count="count")
    .reset_index()
)

_popular_df = movies_df.merge(_rating_stats, on="movieId", how="left")
_popular_df["avg_rating"] = _popular_df["avg_rating"].fillna(0).round(1)
_popular_df["rating_count"] = _popular_df["rating_count"].fillna(0).astype(int)
_popular_df = _popular_df.sort_values(
    ["rating_count", "avg_rating"], ascending=False
)

# 所有出现过的类型（供筛选栏使用）
_all_genres = sorted({
    g for cell in movies_df["genres"].dropna() for g in cell.split("|")
})

PAGE_SIZE = 24


def _prepare(movies, with_details=False):
    """统一补充海报与评分数据，前端只管渲染。"""
    enrich_movies(movies, with_details=with_details)
    for movie in movies:
        if not movie.get("poster"):
            movie["poster"] = get_poster(movie["title"], movie.get("genres", ""))
        avg, count = stats_for(movie["movieId"])
        movie["avg_rating"] = avg
        movie["rating_count"] = count
        movie["year"] = split_title_year(movie["title"])[1]
        genre = primary_genre(movie.get("genres", ""))
        movie["genre_label"] = GENRE_LABELS.get(genre, genre)
    return movies


def _user_rating(movie_id):
    """当前用户对某部电影的评分，未登录/未评分为 None。"""
    if "user_id" not in session:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT rating FROM ratings WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    ).fetchone()
    conn.close()
    return row["rating"] if row else None


def _is_favorite(movie_id):
    if "user_id" not in session:
        return False
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM favorites WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    ).fetchone()
    conn.close()
    return bool(row)


def _is_disliked(movie_id):
    if "user_id" not in session:
        return False
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM dislikes WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    ).fetchone()
    conn.close()
    return bool(row)


def _user_signals():
    """一次性读取当前用户的评分 / 收藏 / 黑名单，供推荐引擎使用。"""
    if "user_id" not in session:
        return {}, [], []
    uid = session["user_id"]
    conn = get_db()
    ratings = {
        row["movie_id"]: row["rating"]
        for row in conn.execute(
            "SELECT movie_id, rating FROM ratings WHERE user_id=?", (uid,)
        ).fetchall()
    }
    favorites = [
        row["movie_id"] for row in conn.execute(
            "SELECT movie_id FROM favorites WHERE user_id=?", (uid,)
        ).fetchall()
    ]
    disliked = [
        row["movie_id"] for row in conn.execute(
            "SELECT movie_id FROM dislikes WHERE user_id=?", (uid,)
        ).fetchall()
    ]
    conn.close()
    return ratings, favorites, disliked


@app.context_processor
def inject_globals():
    return {
        "all_genres": _all_genres,
        "genre_labels": GENRE_LABELS,
        "tmdb_enabled": has_api_key(),
    }


@app.route("/")
def index():
    movies = _popular_df.head(12).to_dict("records")
    _prepare(movies)
    # Hero 轮播：优先选用真实海报的高分影片
    hero_pool = []
    for movie in movies:
        if cached_poster(movie["movieId"]) and movie.get("avg_rating", 0) >= 4:
            hero_pool.append(movie)
        if len(hero_pool) >= 6:
            break
    hero_movies = hero_pool if hero_pool else movies[:6]

    # 登录用户：首页顶部展示个性化推荐（无评分时走冷启动通道）
    personal = []
    if "user_id" in session:
        user_ratings, favorites, disliked = _user_signals()
        if user_ratings:
            personal = recommend_for_user(
                user_ratings, 6, disliked_ids=disliked, favorites=favorites
            )
        else:
            personal = recommend_cold_start(6, disliked_key=tuple(disliked))
        _prepare(personal)

    return render_template(
        "index.html", movies=movies, hero_movies=hero_movies, personal=personal
    )


@app.route("/movies")
def movie_list():
    keyword = request.args.get("keyword", "").strip()
    genre = request.args.get("genre", "").strip()
    sort = request.args.get("sort", "").strip()
    decade = request.args.get("decade", "").strip()  # 如 1990 / 2000 / 2010
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    result = movies_df.copy()

    if keyword:
        result = result[
            result["title"].str.contains(keyword, case=False, na=False)
        ]
    if genre:
        result = result[
            result["genres"].str.contains(genre, case=False, na=False)
        ]
    if decade:
        try:
            d = int(decade)
            # 从片名尾部提取年份，筛选对应年代
            years = result["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
            result = result[(years >= d) & (years < d + 10)]
        except ValueError:
            pass

    if sort in ("popular", "rating"):
        result = result.merge(_rating_stats, on="movieId", how="left")
        result["avg_rating"] = result["avg_rating"].fillna(0)
        result["rating_count"] = result["rating_count"].fillna(0)
        if sort == "rating":
            # 高分优先：评分人数≥20 才参与，避免冷门片虚高
            result = result[result["rating_count"] >= 20]
            result = result.sort_values(["avg_rating", "rating_count"], ascending=False)
        else:
            result = result.sort_values(["rating_count", "avg_rating"], ascending=False)
    elif sort == "year":
        years = result["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
        result = result.assign(_year=years).sort_values("_year", ascending=False)

    total = len(result)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(page, total_pages)
    start = (page - 1) * PAGE_SIZE
    movies = result.iloc[start:start + PAGE_SIZE].to_dict("records")

    # 无排序时用评分人数作为默认热度展示
    for movie in movies:
        avg, count = stats_for(movie["movieId"])
        movie["avg_rating"], movie["rating_count"] = avg, count
    _prepare(movies)

    return render_template(
        "movies.html",
        movies=movies,
        keyword=keyword,
        genre=genre,
        sort=sort,
        decade=decade,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    row = movies_df[movies_df["movieId"] == movie_id]
    if row.empty:
        abort(404)

    movie = row.iloc[0].to_dict()
    _prepare([movie], with_details=True)

    movie["user_rating"] = _user_rating(movie_id)
    movie["is_favorite"] = _is_favorite(movie_id)
    movie["is_disliked"] = _is_disliked(movie_id)
    if not movie.get("tmdb_rating") and movie.get("avg_rating"):
        movie["tmdb_rating"] = movie["avg_rating"]
    # 无 TMDB backdrop 时用真实海报做氛围背景降级
    if not movie.get("backdrop") and movie.get("poster"):
        movie["backdrop"] = movie["poster"]

    rating_dist = rating_distribution(movie_id)
    user_ratings, favorites, disliked = _user_signals() if "user_id" in session else ({}, [], [])
    recommendations = recommend_movies(
        movie_id, 8, disliked_ids=disliked, user_ratings=user_ratings
    )
    _prepare(recommendations)

    return render_template(
        "movie_detail.html",
        movie=movie,
        recommendations=recommendations,
        rating_dist=rating_dist,
    )


@app.route("/search")
def search():
    keyword = request.args.get("keyword", "").strip()
    return redirect(url_for("movie_list", keyword=keyword))


@app.route("/api/suggest")
def api_suggest():
    """搜索建议：返回片名匹配的电影（含海报/评分），供前端自动补全。"""
    keyword = request.args.get("q", "").strip()
    if not keyword:
        return jsonify([])

    matched = movies_df[movies_df["title"].str.contains(keyword, case=False, na=False)]
    # 优先展示评分人数多的（热门），再取 8 条
    matched = matched.merge(_rating_stats, on="movieId", how="left")
    matched["rating_count"] = matched["rating_count"].fillna(0)
    matched = matched.sort_values("rating_count", ascending=False).head(8)

    results = []
    for row in matched.to_dict("records"):
        poster = cached_poster(row["movieId"])
        if not poster:
            poster = get_poster(row["title"], row.get("genres", ""))
        avg, count = stats_for(row["movieId"])
        results.append({
            "id": int(row["movieId"]),
            "title": row["title"],
            "year": split_title_year(row["title"])[1],
            "poster": poster,
            "rating": avg,
        })
    return jsonify(results)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect("/")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        security_answer = request.form.get("security_answer", "").strip()

        if not username:
            return render_template("register.html", error="请输入用户名")
        if len(username) < 3:
            return render_template("register.html", error="用户名至少需要 3 个字符")
        if len(password) < 6:
            return render_template("register.html", error="密码至少需要 6 位")
        if password != confirm_password:
            return render_template("register.html", error="两次输入的密码不一致")
        if not security_answer:
            return render_template("register.html", error="请输入安全问题答案")

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password, security_answer) VALUES (?, ?, ?)",
                (username, generate_password_hash(password),
                 generate_password_hash(security_answer.lower())),
            )
            conn.commit()
        except Exception:
            conn.close()
            return render_template("register.html", error="用户名已经存在，请更换用户名")
        conn.close()

        flash("注册成功，请登录", "success")
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect("/")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return render_template("login.html", error="请输入用户名和密码")

        MAX_ATTEMPTS = 5
        LOCK_SECONDS = 15 * 60

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()

        # 账户锁定检查
        if user and user["locked_until"] and user["locked_until"] > time.time():
            remain = int(user["locked_until"] - time.time())
            conn.close()
            return render_template(
                "login.html",
                error=f"登录失败次数过多，账户已锁定，请 {remain // 60 + 1} 分钟后再试",
            )

        if user and check_password_hash(user["password"], password):
            # 登录成功：重置失败计数
            conn.execute(
                "UPDATE users SET login_attempts=0, locked_until=0 WHERE id=?",
                (user["id"],),
            )
            conn.commit()
            conn.close()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"欢迎回来，{user['username']}！", "success")
            return redirect("/")

        # 登录失败：累加计数，达上限则锁定
        if user:
            attempts = (user["login_attempts"] or 0) + 1
            locked_until = time.time() + LOCK_SECONDS if attempts >= MAX_ATTEMPTS else 0
            conn.execute(
                "UPDATE users SET login_attempts=?, locked_until=? WHERE id=?",
                (attempts, locked_until, user["id"]),
            )
            conn.commit()
            remain = MAX_ATTEMPTS - attempts
            conn.close()
            if remain > 0:
                return render_template(
                    "login.html", error=f"用户名或密码错误，还可尝试 {remain} 次"
                )
            return render_template("login.html", error="账户已锁定 15 分钟，请稍后再试")

        conn.close()
        return render_template("login.html", error="用户名或密码错误")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        security_answer = request.form.get("security_answer", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 6:
            return render_template("forgot_password.html", error="新密码至少需要 6 位")
        if password != confirm_password:
            return render_template("forgot_password.html", error="两次输入的密码不一致")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()

        if not user:
            conn.close()
            return render_template("forgot_password.html", error="用户不存在")
        stored = user["security_answer"] or ""
        # 兼容：新版哈希存储 / 旧版明文存储
        answer_ok = check_password_hash(stored, security_answer)
        legacy_plain = False
        if not answer_ok and stored.lower() == security_answer:
            answer_ok = True
            legacy_plain = True
        if not answer_ok:
            conn.close()
            return render_template("forgot_password.html", error="安全问题答案错误")

        # 旧明文答案顺手升级为哈希
        if legacy_plain:
            conn.execute(
                "UPDATE users SET security_answer=? WHERE id=?",
                (generate_password_hash(security_answer), user["id"]),
            )
        conn.execute(
            "UPDATE users SET password=?, login_attempts=0, locked_until=0 WHERE id=?",
            (generate_password_hash(password), user["id"]),
        )
        conn.commit()
        conn.close()
        return render_template(
            "forgot_password.html", success="密码修改成功，现在可以重新登录"
        )

    return render_template("forgot_password.html")


@app.route("/guest")
def guest():
    session.clear()
    session["guest"] = True
    session["username"] = "游客"
    return redirect("/movies")


@app.route("/logout")
def logout():
    session.clear()
    flash("已安全退出", "info")
    return redirect("/login")


@app.route("/rate/<int:movie_id>", methods=["POST"])
def rate_movie(movie_id):
    if "user_id" not in session:
        return redirect("/login")
    try:
        rating = float(request.form["rating"])
    except (KeyError, ValueError):
        return redirect(f"/movie/{movie_id}")
    if rating < 1 or rating > 5:
        return redirect(f"/movie/{movie_id}")

    conn = get_db()
    conn.execute(
        """INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)
           ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating""",
        (session["user_id"], movie_id, rating),
    )
    conn.commit()
    conn.close()
    flash("评分已保存，感谢你的反馈！", "success")
    return redirect(f"/movie/{movie_id}")


@app.route("/favorite/<int:movie_id>", methods=["POST"])
def add_favorite(movie_id):
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO favorites (user_id, movie_id) VALUES (?, ?)",
        (session["user_id"], movie_id),
    )
    conn.commit()
    conn.close()
    flash("已加入收藏", "success")
    return redirect(f"/movie/{movie_id}")


@app.route("/favorite/remove/<int:movie_id>", methods=["POST"])
def remove_favorite(movie_id):
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute(
        "DELETE FROM favorites WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    )
    conn.commit()
    conn.close()
    flash("已取消收藏", "info")
    return redirect("/favorites")


@app.route("/dislike/<int:movie_id>", methods=["POST"])
def dislike_movie(movie_id):
    """标记不感兴趣：从推荐中剔除，同时若已收藏则一并移除。"""
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO dislikes (user_id, movie_id) VALUES (?, ?)",
        (session["user_id"], movie_id),
    )
    conn.execute(
        "DELETE FROM favorites WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    )
    conn.commit()
    conn.close()
    flash("已标记为不感兴趣，推荐结果会据此调整", "info")
    return redirect(request.referrer or "/recommend")


@app.route("/dislike/remove/<int:movie_id>", methods=["POST"])
def remove_dislike(movie_id):
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute(
        "DELETE FROM dislikes WHERE user_id=? AND movie_id=?",
        (session["user_id"], movie_id),
    )
    conn.commit()
    conn.close()
    flash("已取消标记", "info")
    return redirect(request.referrer or "/recommend")


@app.route("/favorites")
def favorites():
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    rows = conn.execute(
        "SELECT movie_id FROM favorites WHERE user_id=?",
        (session["user_id"],),
    ).fetchall()
    conn.close()

    movie_ids = [row["movie_id"] for row in rows]
    movies = movies_df[movies_df["movieId"].isin(movie_ids)].to_dict("records")
    for movie in movies:
        movie["avg_rating"], movie["rating_count"] = stats_for(movie["movieId"])
    _prepare(movies)
    return render_template("favorites.html", movies=movies)


@app.route("/profile")
def profile():
    """个人中心：评分统计、类型偏好、评分历史。"""
    if "user_id" not in session:
        return redirect("/login")
    uid = session["user_id"]
    conn = get_db()
    rating_rows = conn.execute(
        """SELECT r.movie_id, r.rating, f.id AS fav_id
           FROM ratings r
           LEFT JOIN favorites f ON f.user_id=r.user_id AND f.movie_id=r.movie_id
           WHERE r.user_id=? ORDER BY r.rating DESC, r.movie_id""",
        (uid,),
    ).fetchall()
    fav_count = conn.execute(
        "SELECT COUNT(*) AS c FROM favorites WHERE user_id=?", (uid,)
    ).fetchone()["c"]
    conn.close()

    ratings_list = [dict(row) for row in rating_rows]
    rated_count = len(ratings_list)
    avg_score = (
        round(sum(r["rating"] for r in ratings_list) / rated_count, 2)
        if rated_count else 0
    )

    # 类型偏好：按用户评分过的电影统计类型权重（高评分权重更高）
    genre_weights = {}
    id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))
    for r in ratings_list:
        genres = id_to_genres.get(r["movie_id"], "")
        weight = max(0.0, r["rating"] - 2.5)  # 3分0.5, 4分1.5, 5分2.5
        for g in str(genres).split("|"):
            if g:
                genre_weights[g] = genre_weights.get(g, 0) + weight
    top_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    top_genres = [(GENRE_LABELS.get(g, g), round(w, 1)) for g, w in top_genres]

    # 口味画像一句话总结
    if top_genres:
        g1, g2 = top_genres[0][0], (top_genres[1][0] if len(top_genres) > 1 else "")
        taste_title = f"偏爱{g1}的影迷" if not g2 else f"热爱{g1}与{g2}的影迷"
        if rated_count >= 20:
            taste_title = "资深" + taste_title
        taste_desc = (
            f"从 {rated_count} 条评分中，我们识别出你对「{g1}」"
            + (f"「{g2}」" if g2 else "")
            + " 的明显偏好。推荐系统会据此调整权重，并持续从你的每次评分中学习。"
        )
    else:
        taste_title = "等待发掘的观影新手"
        taste_desc = "多给几部电影评分，我们就能描绘出你的专属口味画像，让推荐越来越精准。"
    # 数据覆盖度（评分数量对推荐精准度的影响）
    coverage = min(100, int(rated_count / 20 * 100))

    # 口味 DNA 雷达图数据
    ratings_dict = {r["movie_id"]: r["rating"] for r in ratings_list}
    dna = taste_dna(ratings_dict, [])

    # 口味 DNA 演化时间线：按评分写入顺序累积，展示口味随评分积累的变化
    conn = get_db()
    timeline_rows = conn.execute(
        "SELECT movie_id, rating FROM ratings WHERE user_id=? ORDER BY id ASC", (uid,)
    ).fetchall()
    conn.close()
    ordered = [(row["movie_id"], row["rating"]) for row in timeline_rows]
    dna_timeline = taste_dna_timeline(ordered)
    dna_labels = [d[0] for d in DNA_DIMS]

    # 社交化推荐：同好也在看
    social = []
    if ratings_dict:
        social = recommend_social(ratings_dict, [], [], 6)
        _prepare(social)

    # 评分历史：补充片名与海报
    history = []
    id_to_movie = movies_df.set_index("movieId").to_dict("index")
    for r in ratings_list[:12]:
        meta = id_to_movie.get(r["movie_id"])
        if not meta:
            continue
        poster = cached_poster(r["movie_id"]) or get_poster(meta["title"], meta.get("genres", ""))
        history.append({
            "movie_id": r["movie_id"],
            "title": meta["title"],
            "poster": poster,
            "rating": r["rating"],
            "is_favorite": bool(r["fav_id"]),
        })

    return render_template(
        "profile.html",
        rated_count=rated_count,
        avg_score=avg_score,
        fav_count=fav_count,
        top_genres=top_genres,
        history=history,
        taste_title=taste_title,
        taste_desc=taste_desc,
        coverage=coverage,
        dna=dna,
        dna_timeline=dna_timeline,
        dna_labels=dna_labels,
        social=social,
    )


@app.route("/recommend")
def recommend():
    if "user_id" not in session:
        return redirect("/login")
    user_ratings, favorites, disliked = _user_signals()
    if user_ratings:
        movies = recommend_for_user(
            user_ratings, 12, disliked_ids=disliked, favorites=favorites
        )
        social = recommend_social(user_ratings, favorites, disliked, 6)
        _prepare(social)
    else:
        movies = recommend_cold_start(12, disliked_key=tuple(disliked))
        social = []
    _prepare(movies)
    return render_template(
        "recommend.html", movies=movies, rated_count=len(user_ratings), social=social
    )


def _chat_user_signals():
    """组装对话引擎需要的用户信号（含高分片名，用于个性化开场白）。"""
    ratings, favorites, disliked = ({}, [], [])
    if "user_id" in session:
        ratings, favorites, disliked = _user_signals()
    top_ids = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:3]
    title_map = dict(zip(movies_df["movieId"], movies_df["title"]))
    top_titles = [str(title_map.get(int(mid), "")) for mid, _ in top_ids if int(mid) in title_map]
    return {
        "ratings": ratings,
        "favorites": favorites,
        "disliked": disliked,
        "top_titles": top_titles,
    }


@app.route("/chat")
def chat():
    """对话式推荐助手：用户用自然语言与 AI 聊天并获得推荐。"""
    return render_template("chat.html", ai_status=chat_engine.ai_status())


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """多轮对话接口：意图识别、会话记忆和片单检索。"""
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "说点什么吧，比如「想看烧脑的科幻片」"})

    signals = _chat_user_signals()
    try:
        result = chat_engine.handle_chat(
            query,
            session,           # 对话上下文存在会话中，支持多轮追问
            user_signals=signals,
            use_llm=True,      # 未配置大模型时引擎内部自动降级
        )
    except Exception as exc:
        app.logger.warning("对话引擎异常，已降级: %s" % exc)
        return jsonify({"ok": False, "error": "这次没太理解，换个说法试试～"})

    # 引擎返回的是原始电影 dict，这里统一补海报再转成前端需要的结构
    raw_movies = result.get("movies") or []
    if raw_movies:
        _prepare(raw_movies)
    result["movies"] = [{
        "id": int(m["movieId"]),
        "title": m["title"],
        "poster": m.get("poster", ""),
        "rating": m.get("avg_rating", 0),
        "reason": m.get("reason", ""),
        "match": m.get("match", 0),
        "genres": m.get("genres", ""),
        "year": split_title_year(m["title"])[1],
    } for m in raw_movies]

    return jsonify(result)


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    """清空当前对话上下文，重新开始一轮新对话。"""
    chat_engine.reset_session(session)
    return jsonify({"ok": True, "message": "已开启新对话"})


@app.route("/mood")
def mood():
    """心情/场景化推荐入口。"""
    mood_key = request.args.get("mood", "").strip()
    user_ratings, favorites, disliked = ({}, [], [])
    if "user_id" in session:
        user_ratings, favorites, disliked = _user_signals()

    movies = []
    if mood_key and mood_key in MOODS:
        movies = recommend_by_mood(
            mood_key, 12, user_ratings=user_ratings,
            disliked_ids=disliked, favorites=favorites,
        )
        _prepare(movies)
    return render_template(
        "mood.html",
        moods=MOODS,
        mood_key=mood_key,
        current=MOODS.get(mood_key),
        movies=movies,
    )


@app.route("/surprise")
def surprise():
    """探索性盲盒：每次换手气给不同片单。"""
    user_ratings, favorites, disliked = ({}, [], [])
    if "user_id" in session:
        user_ratings, favorites, disliked = _user_signals()

    seed = request.args.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (ValueError, TypeError):
        seed = None
    mood = request.args.get("mood", "").strip() or None
    movies = recommend_surprise(
        user_ratings, disliked_ids=disliked, favorites=favorites,
        seed=seed, top_n=8, mood=mood,
    )
    _prepare(movies)
    return render_template(
        "surprise.html", movies=movies, moods=MOODS, mood=mood, seed=seed
    )


@app.route("/report")
def report():
    """年度观影报告：基于用户评分/收藏生成可分享的数据卡片。"""
    if "user_id" not in session:
        return redirect("/login")
    uid = session["user_id"]
    conn = get_db()
    rows = conn.execute(
        "SELECT movie_id, rating FROM ratings WHERE user_id=?", (uid,)
    ).fetchall()
    fav_count = conn.execute(
        "SELECT COUNT(*) AS c FROM favorites WHERE user_id=?", (uid,)
    ).fetchone()["c"]
    dislike_count = conn.execute(
        "SELECT COUNT(*) AS c FROM dislikes WHERE user_id=?", (uid,)
    ).fetchone()["c"]
    conn.close()

    ratings = {row["movie_id"]: row["rating"] for row in rows}
    rated_count = len(ratings)
    if rated_count == 0:
        return render_template(
            "report.html", empty=True, username=session.get("username", "影迷")
        )

    avg_score = round(sum(ratings.values()) / rated_count, 2)
    # 最高评分影片（取最早的）
    top_movies = sorted(ratings.items(), key=lambda x: (-x[1], x[0]))[:5]
    top_list = []
    id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"].fillna("")))
    for mid, score in top_movies:
        meta = movies_df[movies_df["movieId"] == mid]
        if meta.empty:
            continue
        poster = cached_poster(mid) or get_poster(meta.iloc[0]["title"], meta.iloc[0].get("genres", ""))
        top_list.append({
            "movie_id": mid, "title": meta.iloc[0]["title"],
            "poster": poster, "rating": score,
        })

    # 口味关键词：出现最多的类型（按高分加权）
    genre_w = {}
    for mid, score in ratings.items():
        w = max(0.0, score - 2.5)
        for g in str(id_to_genres.get(mid, "")).split("|"):
            if g:
                genre_w[g] = genre_w.get(g, 0) + w
    top_genres = sorted(genre_w.items(), key=lambda x: x[1], reverse=True)[:5]
    keyword_labels = {
        "Drama": "剧情控", "Comedy": "喜剧迷", "Action": "动作派", "Sci-Fi": "科幻粉",
        "Romance": "浪漫派", "Thriller": "悬疑控", "Animation": "童话心", "Crime": "犯罪迷",
        "Adventure": "冒险家", "Horror": "胆大派", "War": "历史控", "Mystery": "推理迷",
    }
    keywords = [keyword_labels.get(g, g) for g, _ in top_genres[:3]]

    dna = taste_dna(ratings, [])
    # 评分分布
    dist = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for score in ratings.values():
        s = min(5, max(1, round(score)))
        dist[s] += 1

    return render_template(
        "report.html",
        empty=False,
        username=session.get("username", "影迷"),
        rated_count=rated_count,
        avg_score=avg_score,
        fav_count=fav_count,
        dislike_count=dislike_count,
        top_list=top_list,
        keywords=keywords,
        dna=dna,
        dist=dist,
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="页面不存在或电影未找到"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="服务器出了点问题，请稍后再试"), 500


if __name__ == "__main__":
    # 本地开发默认 5000；云端发布沙箱通过 PORT 注入端口
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
