"""TMDB 数据层：真正的 API 调用 + 海报路径持久化缓存。

工作方式：
1. 优先从 poster_paths.json 读取已缓存的 poster_path（图片由 TMDB CDN 提供，不落地贴图）。
2. 缓存未命中时，真正调用 TMDB /3/search/movie API 获取 poster_path / 简介 / 评分。
3. API 结果持久化写回 poster_paths.json，下一部同类影片无需重复请求。
4. 只缓存成功结果；失败可自动重试；熔断带冷却自动恢复。

海报图片始终来自 image.tmdb.org CDN（URL = CDN_BASE + poster_path），
因此"海报"是通过 API 读到的路径动态渲染，而非把图片下载到本地贴上去。
"""

import json
import os
import re
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import TMDB_API_KEY, BASE_DIR

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"   # 海报走官方 CDN
PATH_CACHE_FILE = os.path.join(BASE_DIR, "poster_paths.json")
# 详情缓存（简介/氛围图/上映日期/TMDB 评分）。独立于海报缓存，避免破坏已有 1730 条海报。
# 结构: { movieId(str): {"overview": str, "backdrop": str(path|url), "release_date": str,
#                        "vote_average": float|None, "poster": str(path|url), "title": str} }
INFO_CACHE_FILE = os.path.join(BASE_DIR, "movie_info.json")

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MovieRec/1.0)"})

EMPTY_INFO = {
    "poster": None, "overview": "", "release_date": "",
    "vote_average": None, "backdrop": None,
}

_lock = threading.RLock()
_mem_cache = {}          # title -> info（本次进程内，仅成功）
_fetched = set()
_circuit = {"failures": 0, "open": False, "until": 0.0}
_COOLDOWN = 45
_FAILURE_LIMIT = 3


# ---------- 持久化 poster_path 缓存 ----------
def _load_cache():
    try:
        with open(PATH_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_path_cache = _load_cache()   # movieId(str) -> poster_path(str)


def _load_info_cache():
    try:
        with open(INFO_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_info_cache = _load_info_cache()   # movieId(str) -> info dict

# 缓存文件修改时间：抓取脚本在后台持续写入，网站进程据此热重载，无需重启
_cache_mtimes = {
    "path": os.path.getmtime(PATH_CACHE_FILE) if os.path.exists(PATH_CACHE_FILE) else 0.0,
    "info": os.path.getmtime(INFO_CACHE_FILE) if os.path.exists(INFO_CACHE_FILE) else 0.0,
}


def _file_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _refresh_if_changed():
    """若缓存文件被抓取脚本更新，重新载入到内存（带锁，线程安全）。"""
    global _path_cache, _info_cache
    pm = _file_mtime(PATH_CACHE_FILE)
    im = _file_mtime(INFO_CACHE_FILE)
    if pm != _cache_mtimes["path"] or im != _cache_mtimes["info"]:
        with _lock:
            if pm != _cache_mtimes["path"]:
                _path_cache = _load_cache()
                _cache_mtimes["path"] = pm
            if im != _cache_mtimes["info"]:
                _info_cache = _load_info_cache()
                _cache_mtimes["info"] = im


def cached_info(movie_id):
    """读取已缓存的影片详情（简介/氛围图等），无则返回空信息。"""
    _refresh_if_changed()
    data = _info_cache.get(str(movie_id))
    if not data:
        return None
    return {
        "overview": data.get("overview", "") or "",
        "backdrop": _join_image_url(data.get("backdrop")),
        "release_date": data.get("release_date", "") or "",
        "vote_average": data.get("vote_average"),
        "poster": _join_image_url(data.get("poster")),
    }


def info_cache_size():
    return len(_info_cache)


def _save_cache():
    save_cache_if_needed(_path_cache, PATH_CACHE_FILE)


def save_cache_if_needed(cache, path):
    """把 poster_path 缓存原子化写入磁盘（供批量脚本调用）。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def has_api_key():
    return bool(TMDB_API_KEY)


def cache_size():
    return len(_path_cache)


def circuit_open():
    return _circuit["open"] and time.time() < _circuit["until"]


def _record_failure():
    with _lock:
        _circuit["failures"] += 1
        if _circuit["failures"] >= _FAILURE_LIMIT:
            _circuit["open"] = True
            _circuit["until"] = time.time() + _COOLDOWN


def _reset_circuit():
    with _lock:
        _circuit["failures"] = 0
        _circuit["open"] = False


def clean_movie_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title or "").strip()


def extract_year(title):
    m = re.search(r"\((\d{4})\)\s*$", title or "")
    return m.group(1) if m else ""


def _join_image_url(path):
    """把 TMDB poster_path 拼成完整 CDN 图片 URL（兼容带/不带前导斜杠）。"""
    if not path:
        return None
    if path.startswith("http"):
        return path
    return TMDB_IMAGE_BASE + "/" + path.lstrip("/")


def cached_poster(movie_id):
    """若缓存中有该片的 poster_path，返回 TMDB CDN 图片 URL。"""
    _refresh_if_changed()
    path = _path_cache.get(str(movie_id))
    return _join_image_url(path) if path else None


def _path_to_url(path):
    return _join_image_url(path)


def get_movie_info(title, movie_id=None):
    """调用 TMDB API 获取影片信息；成功结果写入内存与持久化缓存。"""
    key = (title or "").strip()

    with _lock:
        if key in _mem_cache:
            return dict(_mem_cache[key])

    if not TMDB_API_KEY or circuit_open():
        return dict(EMPTY_INFO)

    clean_title = clean_movie_title(title)
    year = extract_year(title)

    try:
        resp = _session.get(
            TMDB_SEARCH_URL,
            params={"api_key": TMDB_API_KEY, "query": clean_title,
                    "language": "zh-CN", "year": year},
            timeout=6,
        )
        if resp.status_code == 401:
            with _lock:
                _circuit["open"] = True
                _circuit["until"] = time.time() + 1800
            return dict(EMPTY_INFO)

        results = resp.json().get("results", []) if resp.ok else []
        if not results and year:
            resp = _session.get(
                TMDB_SEARCH_URL,
                params={"api_key": TMDB_API_KEY, "query": clean_title,
                        "language": "zh-CN"},
                timeout=6,
            )
            results = resp.json().get("results", []) if resp.ok else []

        if not results:
            return dict(EMPTY_INFO)

        movie = results[0]
        poster_path = movie.get("poster_path")
        backdrop_path = movie.get("backdrop_path")
        info = {
            "poster": _path_to_url(poster_path),
            "overview": movie.get("overview", "") or "",
            "release_date": movie.get("release_date", "") or "",
            "vote_average": movie.get("vote_average"),
            "backdrop": _path_to_url(backdrop_path),
        }

        with _lock:
            _mem_cache[key] = info
            _fetched.add(key)
            if movie_id is not None and poster_path:
                _path_cache[str(movie_id)] = poster_path
        _save_cache()
        _reset_circuit()
        return dict(info)
    except (requests.RequestException, ValueError):
        _record_failure()
        return dict(EMPTY_INFO)


def enrich_movies(movies, with_details=False):
    """批量补充海报/简介。

    性能优先原则：页面请求只读取本地 poster_paths.json 缓存，命中即用 CDN URL，
    未命中则留空由 poster.py 生成优雅降级海报 —— 绝不在请求中同步调用远程 API，
    否则当 API 被墙时每个页面都会卡住数秒。海报的批量拉取交给 fetch_posters.py 脚本。

    仅当环境变量 TMDB_LIVE_FETCH=1 时才允许运行时实时拉取（适合 API 通畅的部署环境）。
    """
    if not movies:
        return movies

    live = os.environ.get("TMDB_LIVE_FETCH", "0") == "1"

    # 1) 缓存命中：直接用 CDN URL（图片走 CDN，不落地）
    #    同时回填已抓取的详情（简介/氛围图/TMDB评分），来源是网页版详情页解析缓存。
    pending = []
    for movie in movies:
        cached = cached_poster(movie.get("movieId"))
        info = cached_info(movie.get("movieId"))
        if cached:
            movie["poster"] = cached
            movie["_poster_cached"] = True
        elif info and info.get("poster"):
            # 海报缓存缺失但详情缓存里有海报路径，也能用
            movie["poster"] = info["poster"]
            movie["_poster_cached"] = True
        else:
            pending.append(movie)
        # 详情回填（with_details 主要由详情页开启，但简介缓存命中成本低，一并回填）
        if info:
            if info.get("overview"):
                movie["overview"] = info["overview"]
            if info.get("backdrop"):
                movie["backdrop"] = info["backdrop"]
            if info.get("release_date"):
                movie["release_date"] = info["release_date"]
            if info.get("vote_average"):
                movie["tmdb_rating"] = info["vote_average"]

    # 2) 未命中且显式开启实时拉取时，才并发调用 API
    if live and pending and TMDB_API_KEY and not circuit_open():
        seen = set()
        tasks = []
        for m in pending:
            if m["title"] not in seen and m["title"] not in _fetched:
                seen.add(m["title"])
                tasks.append(m)
        if tasks:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(get_movie_info, m["title"], m.get("movieId")): m
                           for m in tasks}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass

    # 3) 回填：重新读缓存（实时拉取成功后会写入缓存）
    for movie in pending:
        cached = cached_poster(movie.get("movieId"))
        if cached:
            movie["poster"] = cached
        else:
            info = _mem_cache.get((movie.get("title") or "").strip())
            if info:
                movie["poster"] = info["poster"]
                if with_details:
                    movie["overview"] = info["overview"]
                    movie["release_date"] = info["release_date"]
                    movie["tmdb_rating"] = info["vote_average"]
                    movie["backdrop"] = info["backdrop"]
    return movies
