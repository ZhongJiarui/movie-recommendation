"""混合推荐引擎 v2

在 v1（仅类型 TF-IDF 相似度 + 热度）基础上升级为：

1. 多路召回：
   - 内容相似度：类型 TF-IDF 余弦（保留 v1 能力，冷启动友好）
   - 隐语义协同：基于全站 10 万条评分做 TruncatedSVD，得到影片/用户口味向量
   - 贝叶斯质量分：样本少的影片向全局均值收缩，避免虚高
   - 热度：log 评分人数，保证推荐可看性
2. 融合排序：各通道在候选池内 min-max 归一化后加权求和
3. MMR 多样性重排：用类型相似度抑制同质化（不再一口气推荐 10 部同类型片）
4. 可解释性：每条推荐附带 reason（推荐理由）与 match（匹配度 0-100）
5. 负反馈：低分（<3）作为反向信号参与口味建模，disliked_id 直接过滤
6. 冷启动：无评分用户走「质量 + 热度 + 多样性」通道

对外保持 v1 的函数签名兼容，新增字段均为附加字段。
"""

import math
import os
from functools import lru_cache

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------- 数据加载
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_df = pd.read_csv(os.path.join(BASE_DIR, "data", "movies.csv"))
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "data", "ratings.csv"))

movies_df["genres_text"] = movies_df["genres"].fillna("").str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df["genres_text"])
# 显式要求稀疏输出，避免 9742x9742 的稠密矩阵占用 ~760MB 内存
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)

# movieId -> DataFrame 行位置，O(1) 查找
_ID_TO_INDEX = {int(mid): idx for idx, mid in enumerate(movies_df["movieId"])}
_INDEX_TO_ID = {idx: int(mid) for mid, idx in _ID_TO_INDEX.items()}
_N_MOVIES = len(movies_df)

# 预计算全站评分统计
_RATING_STATS = (
    ratings_df.groupby("movieId")["rating"]
    .agg(avg_rating="mean", rating_count="count")
)
_RATING_STATS.index = _RATING_STATS.index.astype(int)
_GLOBAL_MEAN = float(ratings_df["rating"].mean())
_MAX_COUNT = float(_RATING_STATS["rating_count"].max()) if len(_RATING_STATS) else 1.0

# 星级分布
_RATING_DIST = {}
for _mid, _grp in ratings_df.groupby("movieId")["rating"]:
    _rounded = _grp.round().clip(1, 5).astype(int)
    _counts = _rounded.value_counts().to_dict()
    _RATING_DIST[int(_mid)] = {s: int(_counts.get(s, 0)) for s in range(1, 6)}

# ---------------------------------------------------------------- 隐语义模型
# 用全站评分构建 user-item 稀疏矩阵，SVD 提取 48 维影片隐向量。
# 用户对影片的口味 = 其评分过的影片隐向量的加权和，低分会得到负权重（负反馈）。
_SVD_DIM = 32
_r = ratings_df[ratings_df["movieId"].isin(_ID_TO_INDEX)]
_u_codes = pd.factorize(_r["userId"])[0]
_i_codes = _r["movieId"].map(_ID_TO_INDEX).to_numpy(dtype=int)
_values = (_r["rating"].to_numpy(dtype=np.float32) - _GLOBAL_MEAN)
_user_item = sparse.csr_matrix(
    (_values, (_u_codes, _i_codes)),
    shape=(int(_u_codes.max()) + 1, _N_MOVIES),
)
_svd = TruncatedSVD(n_components=_SVD_DIM, random_state=42, n_iter=7)
_svd.fit(_user_item)
item_factors = _svd.components_.T.astype(np.float32)          # (n_movies, 48)
_item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
item_factors_n = item_factors / np.maximum(_item_norms, 1e-8)  # 归一化后便于点积当相似度


# ---------------------------------------------------------------- 基础工具
def rating_distribution(movie_id):
    """返回 {1:n,2:n,3:n,4:n,5:n} 星级评分人数分布。"""
    return _RATING_DIST.get(int(movie_id), {s: 0 for s in range(1, 6)})


def stats_for(movie_id):
    """返回 (avg_rating, rating_count)，无评分则为 (0, 0)。"""
    try:
        row = _RATING_STATS.loc[int(movie_id)]
        return round(float(row["avg_rating"]), 1), int(row["rating_count"])
    except KeyError:
        return 0.0, 0


def with_stats(movies):
    """给一批电影 dict 补充 avg_rating / rating_count 字段。"""
    for movie in movies:
        avg, count = stats_for(movie["movieId"])
        movie["avg_rating"] = avg
        movie["rating_count"] = count
    return movies


def _bayesian_score(movie_id, count_prior=20):
    """贝叶斯平滑评分，样本少时向全局均值收缩。"""
    avg, count = stats_for(movie_id)
    if count == 0:
        return 0.0
    return (count * avg + count_prior * _GLOBAL_MEAN) / (count + count_prior)


def _quality_norm(mid):
    return min(_bayesian_score(mid) / 5.0, 1.0)


def _popularity_norm(mid):
    return min(math.log1p(stats_for(mid)[1]) / math.log1p(_MAX_COUNT), 1.0)


def _sim_row(idx):
    """取内容相似度矩阵某一行，返回 1D ndarray（兼容稀疏/稠密）。"""
    row = cosine_sim[idx]
    return row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()


def _content_submatrix(indices):
    sub = cosine_sim[np.ix_(indices, indices)]
    return sub.toarray() if sparse.issparse(sub) else np.asarray(sub)


def _minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def _clean_title(title):
    """去掉尾巴的 (1994) 方便展示。"""
    return str(title).rsplit("(", 1)[0].strip() if "(" in str(title) else str(title)


# ---------------------------------------------------------------- 候选组织
def _candidate_indices(exclude_ids, extra_exclude=None):
    excluded = set(exclude_ids or set()) | set(extra_exclude or set())
    return np.array(
        [i for i, mid in _INDEX_TO_ID.items() if mid not in excluded],
        dtype=int,
    )


def _mmr_select(indices, relevance, sub_sim, top_n, diversity=0.28):
    """最大边缘相关重排：relevance 高、但与已选影片类型过于相似的会被降权。"""
    if len(indices) <= top_n:
        return list(range(len(indices)))

    selected = []
    remaining = list(range(len(indices)))
    lambda_ = 1.0 - diversity
    while remaining and len(selected) < top_n:
        best_pos, best_score = None, -1e9
        for pos in remaining:
            redundancy = (
                float(sub_sim[pos, selected].max()) if selected else 0.0
            )
            score = lambda_ * relevance[pos] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score, best_pos = score, pos
        selected.append(best_pos)
        remaining.remove(best_pos)
    return selected


def _attach_explanations(records, reason_map, match_map):
    for rec in records:
        rec["reason"] = reason_map.get(rec["movieId"], "为你精选")
        rec["match"] = match_map.get(rec["movieId"], 70)
    return records


def _build_result(indices, relevance, sub_sim, top_n, reason_fn, exclude_ids,
                  diversity=0.28):
    """通用：MMR 选片 -> 组装 dict -> 解释与匹配度。"""
    positions = _mmr_select(indices, relevance, sub_sim, top_n, diversity)
    chosen_ids = [int(_INDEX_TO_ID[int(indices[p])]) for p in positions]

    # 匹配度：用融合分在候选池中的百分位映射，保留区分度
    order = np.argsort(np.argsort(relevance))  # 每个候选的排名
    match_map = {}
    reason_map = {}
    for pos, mid in zip(positions, chosen_ids):
        pct = float(order[pos]) / max(len(order) - 1, 1)
        match_map[mid] = int(round(62 + 37 * pct))
        reason_map[mid] = reason_fn(mid, pos, indices, relevance)

    rows = movies_df.iloc[indices[positions]].copy()
    rows["_rank"] = np.arange(len(positions))
    rows = rows.sort_values("_rank")
    result = with_stats(rows.to_dict("records"))
    return _attach_explanations(result, reason_map, match_map)


# ---------------------------------------------------------------- 相似电影（详情页）
def recommend_movies(movie_id, top_n=10, disliked_ids=None, user_ratings=None):
    """基于「隐语义协同 + 类型相似 + 质量」的相似电影推荐。"""
    index = _ID_TO_INDEX.get(int(movie_id))
    if index is None:
        return []

    rated = {int(k) for k in (user_ratings or {}).keys()}
    excluded = rated | set(disliked_ids or set()) | {int(movie_id)}
    candidates = _candidate_indices(excluded)

    # 先用内容相似度圈定同类型候选池，避免冷门片被协同噪声带偏
    content_all = _sim_row(index)
    pool = np.argsort(-content_all)[: max(top_n * 12, 120)]
    pool = np.array([i for i in pool if i not in set(excluded | {index})], dtype=int)
    if len(pool) == 0:
        pool = candidates[:120]

    # 1) 内容：类型余弦（主导信号）
    content = content_all[pool]
    # 2) 隐语义协同：辅助微调
    collab = item_factors_n[pool] @ item_factors_n[index]
    # 3) 质量与热度
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in pool])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in pool])

    relevance = (
        0.58 * _minmax(content)
        + 0.22 * _minmax(collab)
        + 0.14 * _minmax(quality)
        + 0.06 * _minmax(popular)
    )
    candidates, relevance = pool, relevance
    sub_sim = _content_submatrix(candidates)

    def reason_fn(mid, pos, inds, rel):
        seed_title = _clean_title(movies_df.iloc[index]["title"])
        return f"与你看的《{seed_title}》相似"

    return _build_result(candidates, relevance, sub_sim, top_n, reason_fn, excluded)


# ---------------------------------------------------------------- 个性化推荐
def _taste_vectors(user_ratings, favorites=None):
    """由用户评分构造口味向量：正向评分拉近，低分推远；收藏视为强正向。"""
    collab = np.zeros(_SVD_DIM, dtype=np.float32)
    content = np.zeros(tfidf_matrix.shape[1], dtype=np.float32)
    total = 0.0
    signals = []

    for raw_id, raw_rating in user_ratings.items():
        mid = int(raw_id)
        idx = _ID_TO_INDEX.get(mid)
        if idx is None:
            continue
        rating = float(raw_rating)
        w = rating - 2.5  # 5分->2.5, 4分->1.5, 3分->0.5, 2分->-0.5, 1分->-1.5
        if abs(w) < 1e-9:
            continue
        collab += item_factors_n[idx] * w
        row = tfidf_matrix[idx]
        content += (row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()) * w
        total += abs(w)
        if w > 0:
            signals.append(idx)

    # 收藏作为强正反馈（等同于 5 分）
    for raw_id in (favorites or []):
        mid = int(raw_id)
        idx = _ID_TO_INDEX.get(mid)
        if idx is None or idx in signals:
            continue
        collab += item_factors_n[idx] * 2.5
        row = tfidf_matrix[idx]
        content += (row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()) * 2.5
        total += 2.5
        signals.append(idx)

    if total > 0:
        collab /= total
        content /= total
    return collab, content, signals


def _recommend_core(user_ratings, top_n, disliked_ids, favorites, diversity):
    ratings_key = tuple(sorted((int(k), round(float(v), 1)) for k, v in user_ratings.items()))
    return _recommend_cached(ratings_key, top_n, tuple(sorted(disliked_ids or set())),
                             tuple(sorted(int(x) for x in (favorites or []))), diversity)


@lru_cache(maxsize=128)
def _recommend_cached(ratings_key, top_n, disliked_key, favorites_key, diversity):
    user_ratings = dict(ratings_key)
    disliked = set(disliked_key)
    favorites = set(favorites_key)

    collab_vec, content_vec, signals = _taste_vectors(user_ratings, favorites)
    if not signals:
        return []

    rated = set(user_ratings.keys()) | favorites
    candidates = _candidate_indices(rated | disliked)

    # 1) 隐语义协同得分
    collab_scores = item_factors_n[candidates] @ collab_vec
    # 2) 内容得分
    content_scores = tfidf_matrix[candidates] @ content_vec
    content_scores = np.asarray(content_scores).ravel()
    # 3) 质量 + 热度
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in candidates])

    # 冷启动（评分很少）时降低协同权重，提高质量与热度占比
    n_signals = len(signals)
    if n_signals <= 2:
        w = (0.28, 0.24, 0.30, 0.18)
    else:
        w = (0.44, 0.18, 0.26, 0.12)

    relevance = (
        w[0] * _minmax(collab_scores)
        + w[1] * _minmax(content_scores)
        + w[2] * _minmax(quality)
        + w[3] * _minmax(popular)
    )

    # 只取融合分前 220 部做 MMR
    keep = np.argsort(-relevance)[:220]
    candidates, relevance = candidates[keep], relevance[keep]
    sub_sim = _content_submatrix(candidates)

    def reason_fn(mid, pos, inds, rel):
        # 找到与它最像的一条正向信号
        vec = item_factors_n[_ID_TO_INDEX[mid]]
        sims = item_factors_n[signals] @ vec
        best = signals[int(np.argmax(sims))]
        seed = _clean_title(movies_df.iloc[best]["title"])
        avg, count = stats_for(mid)
        if count < 25:
            return f"与《{seed}》口味相近 · 小众佳作"
        if avg >= 4.2:
            return f"因《{seed}》而推荐 · 高分口碑"
        return f"与你看过的《{seed}》相似"

    return _build_result(candidates, relevance, sub_sim, top_n, reason_fn,
                         rated | disliked, diversity)


def recommend_for_user(user_ratings, top_n=10, disliked_ids=None, favorites=None,
                       diversity=0.28):
    """个性化推荐：口味画像 + 质量热度，负反馈参与建模，MMR 保证多样性。"""
    if not user_ratings:
        return []
    return _recommend_core(user_ratings, top_n, disliked_ids, favorites, diversity)


@lru_cache(maxsize=32)
def recommend_cold_start(top_n=12, disliked_key=()):
    """冷启动：无评分用户的热门高质量推荐（带类型多样性）。"""
    disliked = set(disliked_key)
    candidates = _candidate_indices(disliked)
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    relevance = 0.62 * _minmax(quality) + 0.38 * _minmax(popular)
    keep = np.argsort(-relevance)[:220]
    candidates, relevance = candidates[keep], relevance[keep]
    sub_sim = _content_submatrix(candidates)

    def reason_fn(mid, pos, inds, rel):
        avg, count = stats_for(mid)
        return f"影迷高分推荐 · {count} 人评分" if count >= 100 else "值得发现的好电影"

    return _build_result(candidates, relevance, sub_sim, top_n, reason_fn,
                         disliked, diversity=0.32)


# ========================================================================
# 扩展能力：心情场景 / 盲盒 / 社交同好 / 口味 DNA
# ========================================================================

# 社交推荐需要「用户隐向量」：SVD.transform 得到 610 个虚拟用户在 32 维空间的位置
user_factors = _svd.transform(_user_item)                 # (610, 32)
# 每个虚拟用户的高分影片（评分 >= 4，即中心化后 >= 0.4）
_pos_mask = (_user_item.data >= 0.4).astype(np.float32)
_user_pos = sparse.csr_matrix(
    (_pos_mask, _user_item.indices, _user_item.indptr), shape=_user_item.shape
)

# 类型 -> 影片行索引，便于按类型快速打分
_GENRE_TO_INDICES = {}
for _i, _cell in enumerate(movies_df["genres"].fillna("")):
    for _g in str(_cell).split("|"):
        if _g:
            _GENRE_TO_INDICES.setdefault(_g, []).append(_i)


# 心情场景配置：每种情绪映射到一组类型权重
MOODS = {
    "laugh": {
        "name": "今晚想笑一场", "emoji": "😂", "desc": "轻松喜剧，卸下一天的疲惫",
        "genres": {"Comedy": 3.0, "Animation": 2.0, "Children": 1.0},
        "min_rating": 3.6,
    },
    "brain": {
        "name": "烧脑悬疑夜", "emoji": "🧠", "desc": "反转再反转，让大脑高速运转",
        "genres": {"Mystery": 3.0, "Thriller": 2.5, "Sci-Fi": 1.5, "Crime": 1.0},
        "min_rating": 3.7,
    },
    "romance": {
        "name": "浪漫时刻", "emoji": "💜", "desc": "心动与温情，适合两个人一起看",
        "genres": {"Romance": 3.0, "Drama": 1.2, "Musical": 1.0},
        "min_rating": 3.6,
    },
    "hero": {
        "name": "热血沸腾", "emoji": "🔥", "desc": "肾上腺素拉满的冒险之旅",
        "genres": {"Action": 3.0, "Adventure": 2.5, "War": 1.0},
        "min_rating": 3.5,
    },
    "heal": {
        "name": "治愈温暖", "emoji": "🌿", "desc": "柔软又明亮，给心情充个电",
        "genres": {"Animation": 2.5, "Children": 2.0, "Drama": 1.2, "Comedy": 1.0},
        "min_rating": 3.7,
    },
    "deep": {
        "name": "深夜沉思", "emoji": "🌙", "desc": "安静深刻，和自己的内心对话",
        "genres": {"Drama": 3.0, "Film-Noir": 1.5, "Mystery": 1.0},
        "min_rating": 3.8,
    },
}


def _genre_scores(cfg):
    """计算每部影片对某心情的匹配分（0~1）。"""
    scores = np.zeros(_N_MOVIES, dtype=np.float32)
    total_w = float(sum(cfg["genres"].values()))
    for genre, weight in cfg["genres"].items():
        for i in _GENRE_TO_INDICES.get(genre, []):
            scores[i] += weight
    if total_w > 0:
        scores /= total_w
    return scores


@lru_cache(maxsize=32)
def _mood_cached(mood_key, top_n, disliked_key, rated_key, fav_key):
    cfg = MOODS.get(mood_key, MOODS["laugh"])
    disliked = set(disliked_key) | set(rated_key) | set(fav_key)
    candidates = _candidate_indices(disliked)

    genre = _genre_scores(cfg)[candidates]
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in candidates])

    # 只保留达到该心情质量门槛的影片
    min_q = cfg["min_rating"] / 5.0
    mask = quality >= min_q
    if mask.sum() < top_n * 5:
        mask = quality >= (cfg["min_rating"] - 0.4) / 5.0
    pool = candidates[mask]
    if len(pool) == 0:
        pool = candidates

    genre = _genre_scores(cfg)[pool]
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in pool])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in pool])
    relevance = 0.62 * _minmax(genre) + 0.28 * _minmax(quality) + 0.10 * _minmax(popular)
    keep = np.argsort(-relevance)[:220]
    pool, relevance = pool[keep], relevance[keep]
    sub_sim = _content_submatrix(pool)

    def reason_fn(mid, pos, inds, rel):
        return f"{cfg['emoji']} {cfg['name']} · 口碑 {stats_for(mid)[0]} 分"

    return _build_result(pool, relevance, sub_sim, top_n, reason_fn, disliked, diversity=0.3)


def recommend_by_mood(mood_key, top_n=12, user_ratings=None, disliked_ids=None, favorites=None):
    """心情场景化推荐：按情绪对应的类型权重 + 质量门槛生成片单。"""
    rated = tuple(sorted(int(k) for k in (user_ratings or {}).keys()))
    disliked = tuple(sorted(int(k) for k in (disliked_ids or [])))
    fav = tuple(sorted(int(k) for k in (favorites or [])))
    return _mood_cached(mood_key, top_n, disliked, rated, fav)


def recommend_surprise(user_ratings, disliked_ids=None, favorites=None, seed=None,
                       top_n=8, mood=None, min_rating=3.7):
    """探索性盲盒：口味匹配 + 高分中长尾 + 随机抽样，每次都不一样。"""
    rng = np.random.default_rng(seed)
    collab_vec, content_vec, signals = _taste_vectors(user_ratings, favorites)
    excluded = set(user_ratings.keys()) | set(disliked_ids or []) | set(favorites or [])
    candidates = _candidate_indices(excluded)

    if signals:
        collab_scores = item_factors_n[candidates] @ collab_vec
    else:
        collab_scores = _popularity_norm_array(candidates)

    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    popular = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in candidates])

    # 质量门槛 + 中长尾偏好（越冷门越有探索价值）
    novelty = 1.0 - popular
    gate = quality >= min_rating / 5.0
    if gate.sum() < 300:
        gate = quality >= (min_rating - 0.5) / 5.0
    pool = candidates[gate]
    if len(pool) < top_n:
        pool = candidates

    c = item_factors_n[pool] @ collab_vec if signals else np.zeros(len(pool))
    q = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in pool])
    p = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in pool])
    relevance = 0.38 * _minmax(c) + 0.34 * _minmax(q) + 0.28 * _minmax(1.0 - p)

    # 心情约束：在候选中叠加类型偏好
    if mood and mood in MOODS:
        relevance = relevance + 0.35 * _minmax(_genre_scores(MOODS[mood])[pool])

    pool_size = min(len(pool), max(top_n * 18, 90))
    shortlist = np.argsort(-relevance)[:pool_size]
    if len(shortlist) > top_n:
        pick = rng.choice(shortlist, size=top_n, replace=False)
    else:
        pick = shortlist
    # 按相关度排序但保留随机选中的影片
    pick = pick[np.argsort(-relevance[pick])]
    chosen_pool = pool[pick]
    chosen_rel = relevance[pick]

    sub_sim = _content_submatrix(chosen_pool)
    reason_map = {}
    match_map = {}
    for k, pos in enumerate(range(len(chosen_pool))):
        mid = int(_INDEX_TO_ID[int(chosen_pool[pos])])
        reason_map[mid] = "🎁 盲盒惊喜 · 高分冷门好片" if stats_for(mid)[1] < 120 else "🎁 盲盒惊喜 · 值得一看"
        match_map[mid] = int(round(70 + 29 * (1 - k / max(len(chosen_pool) - 1, 1))))
    rows = movies_df.iloc[chosen_pool].copy()
    rows["_rank"] = np.arange(len(chosen_pool))
    result = with_stats(rows.sort_values("_rank").to_dict("records"))
    return _attach_explanations(result, reason_map, match_map)


def _popularity_norm_array(indices):
    return np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in indices])


def recommend_social(user_ratings, favorites=None, disliked_ids=None, top_n=10):
    """社交化推荐：SVD 空间里找口味最像的虚拟影迷，推荐他们高分看过、你没看过的片。"""
    if not user_ratings and not favorites:
        return []

    # 把当前用户 fold-in 到与虚拟用户相同的隐语义空间：用户因子 = 评分向量 @ 物品因子
    r = np.zeros(_N_MOVIES, dtype=np.float32)
    for mid, rating in user_ratings.items():
        idx = _ID_TO_INDEX.get(int(mid))
        if idx is not None:
            r[idx] = float(rating) - _GLOBAL_MEAN
    for mid in (favorites or []):
        idx = _ID_TO_INDEX.get(int(mid))
        if idx is not None:
            r[idx] = max(r[idx], 1.5)
    app_vec = r @ item_factors                                    # (32,)
    sims = cosine_similarity(app_vec.reshape(1, -1), user_factors).ravel()
    top_users = np.argsort(-sims)[:40]
    weights = np.clip(sims[top_users], 0, None)
    if weights.sum() <= 1e-8:
        return recommend_for_user(user_ratings, top_n, disliked_ids, favorites)

    # 同好的高分影片按相似度加权聚合
    friend_scores = np.asarray((weights @ _user_pos[top_users]).ravel())
    excluded = set(user_ratings.keys()) | set(disliked_ids or []) | set(favorites or [])
    candidates = _candidate_indices(excluded)

    scores = friend_scores[candidates]
    quality = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    relevance = 0.68 * _minmax(scores) + 0.32 * _minmax(quality)
    keep = np.argsort(-relevance)[:220]
    pool, relevance = candidates[keep], relevance[keep]
    sub_sim = _content_submatrix(pool)
    n_friends = int((weights > 0.25).sum())

    def reason_fn(mid, pos, inds, rel):
        return f"👥 {max(n_friends, 3)} 位和你口味相近的影迷高分推荐"

    return _build_result(pool, relevance, sub_sim, top_n, reason_fn, excluded, diversity=0.3)


# 口味 DNA 维度：中文维度名 -> 对应 MovieLens 类型
DNA_DIMS = [
    ("喜剧", ["Comedy"]),
    ("动作", ["Action", "Adventure"]),
    ("剧情", ["Drama"]),
    ("科幻", ["Sci-Fi", "Fantasy"]),
    ("爱情", ["Romance"]),
    ("悬疑", ["Thriller", "Mystery", "Crime", "Film-Noir"]),
]


def taste_dna(user_ratings, favorites=None):
    """口味 DNA：6 维类型偏好强度（0~100），用于雷达图。"""
    raw = {name: 0.0 for name, _ in DNA_DIMS}
    all_signals = list(user_ratings.items())
    all_signals += [(mid, 5.0) for mid in (favorites or [])]
    id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"].fillna("")))

    for mid, rating in all_signals:
        w = max(0.0, float(rating) - 2.5)
        if w <= 0:
            continue
        genres = str(id_to_genres.get(int(mid), "")).split("|")
        for name, members in DNA_DIMS:
            if any(g in members for g in genres):
                raw[name] += w
    peak = max(raw.values()) if raw and max(raw.values()) > 0 else 1.0
    return [(name, int(round(v / peak * 100))) for name, v in raw.items()]


# ========================================================================
# 对话式推荐：自然语言解析 + 意图融合
# ========================================================================
import re as _re

# 常见片名中文别名 -> 数据集片名关键词，让用户可以用中文指代参考片
CHAT_TITLE_ALIASES = {
    "星际穿越": "Interstellar", "盗梦空间": "Inception",
    "肖申克的救赎": "Shawshank", "肖申克": "Shawshank",
    "玩具总动员": "Toy Story", "黑客帝国": "Matrix",
    "阿甘正传": "Forrest Gump", "泰坦尼克": "Titanic",
    "搏击俱乐部": "Fight Club", "黑暗骑士": "Dark Knight",
    "低俗小说": "Pulp Fiction", "教父": "Godfather",
    "辛德勒的名单": "Schindler", "美丽人生": "Life Is Beautiful",
    "千与千寻": "Spirited Away", "龙猫": "My Neighbor Totoro",
    "天空之城": "Castle in the Sky", "三傻大闹宝莱坞": "Three Idiots",
    "这个杀手不太冷": "Professional", "霸王别姬": "Farewell My Concubine",
    "哈利波特": "Harry Potter", "指环王": "Lord of the Rings", "魔戒": "Lord of the Rings",
    "星球大战": "Star Wars", "复仇者联盟": "Avengers", "钢铁侠": "Iron Man",
    "蜘蛛侠": "Spider-Man", "蝙蝠侠": "Batman", "异形": "Alien",
    "终结者": "Terminator", "回到未来": "Back to the Future",
    "海底总动员": "Finding Nemo", "狮子王": "Lion King",
    "美女与野兽": "Beauty and the Beast", "阿拉丁": "Aladdin",
    "花木兰": "Mulan", "飞屋环游记": "Up", "怪物公司": "Monsters",
}

# 心情关键词 -> MOODS key
CHAT_MOOD_KEYWORDS = {
    "laugh": ["笑", "喜剧", "搞笑", "开心", "轻松", "欢乐", "爆笑", "愉快", "快乐"],
    "brain": ["烧脑", "悬疑", "推理", "反转", "解谜", "智力", "谜题", "惊悚"],
    "romance": ["浪漫", "爱情", "恋爱", "心动", "告白", "情侣", "甜蜜", "温情"],
    "hero": ["热血", "刺激", "动作", "冒险", "燃", "爆炸", "英雄", "爽"],
    "heal": ["治愈", "温暖", "感动", "温馨", "疗愈", "暖心", "治愈系"],
    "deep": ["深夜", "沉思", "深刻", "文艺", "安静", "孤独", "思考人生"],
}

# 中文类型关键词 -> MovieLens 类型
CHAT_GENRE_KEYWORDS = {
    "科幻": "Sci-Fi", "未来": "Sci-Fi", "太空": "Sci-Fi",
    "动画": "Animation", "卡通": "Animation",
    "恐怖": "Horror", "惊悚": "Horror",
    "爱情": "Romance", "恋爱": "Romance",
    "喜剧": "Comedy", "搞笑": "Comedy",
    "动作": "Action", "打斗": "Action",
    "悬疑": "Mystery", "推理": "Mystery",
    "犯罪": "Crime", "战争": "War", "西部": "Western",
    "音乐": "Musical", "家庭": "Children", "奇幻": "Fantasy", "魔幻": "Fantasy",
    "纪录": "Documentary",
}

_QUALITY_WORDS = ["高分", "口碑", "经典", "评分高", "最好", "佳作", "神作", "精品", "豆瓣高分"]


def _resolve_title(keyword):
    """按片名关键词找一部电影，优先评分人数多的，返回 (idx, title)。"""
    hit = movies_df[movies_df["title"].str.contains(keyword, case=False, na=False, regex=False)]
    if hit.empty:
        return None, None
    counts = hit["movieId"].map(lambda m: stats_for(int(m))[1])
    top = hit.assign(_cnt=counts).sort_values("_cnt", ascending=False).iloc[0]
    return int(_ID_TO_INDEX[int(top["movieId"])]), top["title"]


def parse_chat_query(query):
    """把一句自然语言解析为推荐意图 dict（纯规则，无外部依赖）。"""
    query = (query or "").strip()
    ref_idx, ref_title = None, None

    # 1) 中文别名
    if ref_idx is None:
        for alias, kw in CHAT_TITLE_ALIASES.items():
            if alias in query:
                ref_idx, ref_title = _resolve_title(kw)
                if ref_idx is not None:
                    break
    # 2) 书名号《xxx》
    if ref_idx is None:
        for inner in _re.findall(r"《([^》]+)》", query):
            kw = CHAT_TITLE_ALIASES.get(inner, inner)
            ref_idx, ref_title = _resolve_title(kw)
            if ref_idx is not None:
                break
    # 3) 英文片名词（长度>=4），与片名匹配
    if ref_idx is None:
        for word in _re.findall(r"[A-Za-z][A-Za-z' .-]{3,}", query):
            word = word.strip()
            if len(word) < 4 or word.lower() in ("like", "with", "that", "some", "good", "best"):
                continue
            ref_idx, ref_title = _resolve_title(word)
            if ref_idx is not None:
                break

    # 心情
    mood_key = None
    for key, words in CHAT_MOOD_KEYWORDS.items():
        if any(w in query for w in words):
            mood_key = key
            break
    # 否定语义：包含"不要太X / 没那么X / 不想看X"时，撤销对应心情
    _NEG_PATTERNS = ["不要太", "别太", "没那么", "不想太", "不要那么", "不想看", "别搞"]
    if mood_key and any(p in query for p in _NEG_PATTERNS):
        mood_key = None

    # 类型
    genre_weights, genre_cn = {}, []
    for cn, eng in CHAT_GENRE_KEYWORDS.items():
        if cn in query:
            genre_weights[eng] = 1.0
            genre_cn.append(cn)
    # 心情若命中，也把其类型并入（增强信号）
    if mood_key:
        for g, w in MOODS[mood_key]["genres"].items():
            genre_weights[g] = max(genre_weights.get(g, 0), w)

    # 年代：1990年代 / 90年代 / 2010年代
    decade = None
    m = _re.search(r"(19\d{2}|20\d{2})\s*年代", query)
    if m:
        decade = int(m.group(1))
    else:
        m = _re.search(r"(\d{2})\s*年代", query)
        if m:
            yy = int(m.group(1))
            decade = 1900 + yy if yy >= 30 else 2000 + yy
    if "千禧" in query or "零零年代" in query:
        decade = 2000

    quality = any(w in query for w in _QUALITY_WORDS)

    # 生成人话摘要与标签
    summary_parts = []
    if ref_title:
        summary_parts.append(f"和《{_clean_title(ref_title)}》相似")
    if mood_key:
        summary_parts.append(MOODS[mood_key]["name"])
    if genre_cn:
        summary_parts.append("、".join(genre_cn[:2]) + "类")
    if decade:
        summary_parts.append(f"{decade}年代")
    if quality:
        summary_parts.append("高口碑")
    summary = "想找" + "、".join(summary_parts) + "的电影" if summary_parts else "为你精挑细选的好电影"

    tags = []
    if ref_title:
        tags.append(f"参考《{_clean_title(ref_title)}》")
    if mood_key:
        tags.append(MOODS[mood_key]["emoji"] + " " + MOODS[mood_key]["name"])
    for cn in genre_cn[:2]:
        tags.append(cn + "片")
    if decade:
        tags.append(f"{decade}年代")
    if quality:
        tags.append("⭐ 高分口碑")

    return {
        "reference_idx": ref_idx, "reference_title": ref_title,
        "mood": mood_key, "genres": genre_weights, "genre_cn": "、".join(genre_cn),
        "decade": decade, "quality": quality,
        "summary": summary, "tags": tags, "raw": query,
    }


def recommend_by_chat(query, top_n=8, user_ratings=None, disliked_ids=None, favorites=None):
    """对话式推荐：解析自然语言意图，融合参考片/心情/类型/年代/口碑生成片单。"""
    intent = parse_chat_query(query)
    excluded = set(int(k) for k in (user_ratings or {}).keys())
    excluded |= set(int(k) for k in (disliked_ids or []))
    excluded |= set(int(k) for k in (favorites or []))
    ref_idx = intent["reference_idx"]
    if ref_idx is not None:
        excluded.add(_INDEX_TO_ID[ref_idx])  # 排除的是 movieId，不是 DataFrame 行索引
    candidates = _candidate_indices(excluded)

    # 年代过滤（候选足够时才启用，避免结果过少）
    if intent["decade"] and len(candidates) > top_n * 6:
        years = movies_df.iloc[candidates]["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
        mask = ((years >= intent["decade"]) & (years < intent["decade"] + 10)).to_numpy()
        if mask.sum() >= top_n * 3:
            candidates = candidates[mask]

    collab_vec, content_vec, signals = _taste_vectors(user_ratings, favorites)

    components = []  # (weight, score_vector)

    # 参考片：内容相似为主 + 隐语义协同为辅
    if ref_idx is not None:
        c = _sim_row(ref_idx)[candidates]
        cf = item_factors_n[candidates] @ item_factors_n[ref_idx]
        components.append((0.42, 0.72 * _minmax(c) + 0.28 * _minmax(cf)))
    # 用户口味
    if signals:
        cv = item_factors_n[candidates] @ collab_vec
        ct = np.asarray(tfidf_matrix[candidates] @ content_vec).ravel()
        components.append((0.34, 0.62 * _minmax(cv) + 0.38 * _minmax(ct)))
    # 心情 / 类型
    if intent["genres"]:
        g = _genre_scores({"genres": intent["genres"]})[candidates]
        components.append((0.30, _minmax(g)))
    # 质量 + 热度
    q = np.array([_quality_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    p = np.array([_popularity_norm(_INDEX_TO_ID[int(i)]) for i in candidates])
    qw = 0.34 if intent["quality"] else 0.20
    components.append((qw, 0.76 * _minmax(q) + 0.24 * _minmax(p)))

    if not components:
        return intent, recommend_cold_start(top_n, disliked_key=tuple(int(x) for x in (disliked_ids or [])))

    total_w = sum(w for w, _ in components)
    relevance = sum(w / total_w * s for w, s in components)

    keep = np.argsort(-relevance)[:220]
    candidates, relevance = candidates[keep], relevance[keep]
    sub_sim = _content_submatrix(candidates)

    genre_cn = intent["genre_cn"]

    def reason_fn(mid, pos, inds, rel):
        bits = []
        if ref_idx is not None:
            bits.append(f"类似《{_clean_title(movies_df.iloc[ref_idx]['title'])}》")
        if intent["mood"]:
            bits.append(MOODS[intent["mood"]]["name"])
        if genre_cn:
            bits.append(genre_cn.split("、")[0] + "向")
        avg, cnt = stats_for(mid)
        if avg >= 4.2 and cnt >= 100:
            bits.append("高分口碑")
        elif cnt < 60:
            bits.append("小众佳片")
        return " · ".join(bits) if bits else "为你精选"

    movies = _build_result(candidates, relevance, sub_sim, top_n, reason_fn, excluded, diversity=0.3)
    return intent, movies


def taste_dna_timeline(ordered_ratings, favorites=None, max_steps=6):
    """口味 DNA 演化：把评分按时间顺序逐步累积，生成若干快照，展示口味画像如何演变。

    ordered_ratings: [(movie_id, rating), ...] 按时间先后排列（无时间戳时退化为评分写入顺序）
    返回 [{"label": "第N部", "count": N, "dna": [[维度, 强度], ...]}, ...]
    """
    ordered = [(int(mid), float(r)) for mid, r in ordered_ratings]
    n = len(ordered)
    if n == 0:
        return []

    # 快照节点：至少取 2 个（含最终），最多 max_steps 个
    if n <= 2:
        steps = list(range(1, n + 1))
    else:
        raw = np.linspace(1, n, max_steps)
        steps = sorted(set(int(round(s)) for s in raw))
        if steps[-1] != n:
            steps.append(n)
        if len(steps) < 2:
            steps = [max(1, n - 1), n]

    fav_ids = set(int(x) for x in (favorites or []))
    timeline = []
    for k in steps:
        subset = dict(ordered[:k])
        fav_so_far = [m for m in fav_ids if m in subset]
        dna = taste_dna(subset, fav_so_far)
        timeline.append({
            "label": f"第 {k} 部",
            "count": k,
            "dna": dna,
        })
    return timeline
