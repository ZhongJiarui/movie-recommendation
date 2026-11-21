"""批量拉取电影海报 + 中文剧情简介（API / 网页版双通道，自动切换）。

背景：在中国大陆 api.themoviedb.org 常被阻断，但 image.tmdb.org 图片 CDN
和 www.themoviedb.org 网页版通常可直连。本脚本启动时会自检：
  - API 通 → 走官方 API（快、中文信息准）
  - API 不通 → 自动降级到网页版搜索（无需开加速器，效果相同）

网页版通道的简介来源：搜索结果里拿到 /movie/{id-slug}，再请求该详情页，
从 <meta name="description"> 提取中文剧情简介，从 og:image 提取氛围大图。

缓存写入（两个独立文件，互不破坏）：
  - poster_paths.json : movieId -> poster_path（兼容旧格式，海报图片走 CDN）
  - movie_info.json   : movieId -> {overview, backdrop, release_date, vote_average, poster, title}

用法：
    python fetch_posters.py                 # 自动选通道，拉全部
    python fetch_posters.py --limit 500     # 先拉热度前 500 部
    python fetch_posters.py --workers 8     # 并发线程数（网页版建议 ≤6）
    python fetch_posters.py --channel web   # 强制走网页版（免加速）
    python fetch_posters.py --channel api   # 强制走官方 API
"""

import argparse
import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from config import TMDB_API_KEY, TMDB_BEARER_TOKEN, BASE_DIR
from tmdb import (
    PATH_CACHE_FILE, INFO_CACHE_FILE,
    _path_cache, save_cache_if_needed,
    extract_year, _join_image_url,
)

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_DETAIL_URL = "https://api.themoviedb.org/3/movie/{id}"
WEB_SEARCH_URL = "https://www.themoviedb.org/search/movie"
WEB_DETAIL_URL = "https://www.themoviedb.org/movie/{id}"

# 直连（不走环境变量代理）—— 针对国内环境：TMDB 网页版用直连反而通
DIRECT = {"http": None, "https": None}
ZH_HEADERS = {"Accept-Language": "zh-CN,zh;q=0.9"}

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MovieRec/1.0)"})
_web_session = requests.Session()  # 网页版不要自定义 UA，否则会 403

_print_lock = threading.Lock()
_cache_lock = threading.Lock()
_stats = {"ok": 0, "none": 0, "fail": 0, "skip": 0, "info": 0}
_fail_samples = []       # 记录前几条失败样本，便于诊断
_CHANNEL = "api"         # 全局通道标记
_FILL_MISSING = False    # 补漏模式：只补简介为空的影片

# 详情缓存（本进程内），结构同 tmdb.INFO_CACHE_FILE
_info_cache = {}
try:
    with open(INFO_CACHE_FILE, "r", encoding="utf-8") as _f:
        _loaded = json.load(_f)
        if isinstance(_loaded, dict):
            _info_cache = _loaded
except (OSError, ValueError):
    _info_cache = {}


def clean_movie_title(title):
    """清洗 MovieLens 片名，提升搜索准确率。"""
    c = re.sub(r"\s*\(\d{4}\)\s*$", "", title or "").strip()
    m = re.search(r"a\.k\.a\.\s*(.+?)\s*\)", c)
    if m:
        c = m.group(1).strip()
    c = re.sub(r"^(.+),\s*(The|A|An)$", r"\2 \1", c, flags=re.I)
    c = re.sub(r"\s*\([^)]*\)\s*", " ", c).strip()
    return re.sub(r"\s+", " ", c)


# ---------- 通道自检 ----------
def _api_auth():
    if TMDB_BEARER_TOKEN:
        return {}, {"Authorization": f"Bearer {TMDB_BEARER_TOKEN}"}
    return {"api_key": TMDB_API_KEY}, {}


def probe_api(timeout=6):
    if not (TMDB_API_KEY or TMDB_BEARER_TOKEN):
        return False, "未配置密钥或令牌"
    params, headers = _api_auth()
    params["query"] = "Toy Story"
    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, headers=headers, timeout=timeout)
        if r.status_code == 401:
            return False, "密钥无效(401)"
        if r.ok and "results" in r.json():
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, type(e).__name__


def probe_web(timeout=8):
    try:
        r = _web_session.get(WEB_SEARCH_URL, params={"query": "Toy Story"},
                             timeout=timeout, proxies=DIRECT)
        if r.ok and "/movie/" in r.text:
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, type(e).__name__


# ---------- 单部查询 ----------
def _fetch_via_api(title):
    """官方 API 通道：search 拿 poster_path + id，再 detail 拿中文简介。"""
    year = extract_year(title)
    auth_params, auth_headers = _api_auth()
    params = {**auth_params, "query": clean_movie_title(title), "language": "zh-CN"}
    if year:
        params["year"] = year
    resp = _session.get(TMDB_SEARCH_URL, params=params, headers=auth_headers, timeout=10)
    results = resp.json().get("results", []) if resp.ok else []
    if not results and year:
        params.pop("year", None)
        resp = _session.get(TMDB_SEARCH_URL, params=params, headers=auth_headers, timeout=10)
        results = resp.json().get("results", []) if resp.ok else []
    if not results:
        return None
    hit = results[0]
    info = {
        "poster": hit.get("poster_path"),
        "title": hit.get("title", ""),
        "overview": "",
        "backdrop": None,
        "release_date": hit.get("release_date", "") or "",
        "vote_average": hit.get("vote_average"),
    }
    # 详情接口取中文简介与 backdrop
    tid = hit.get("id")
    if tid:
        try:
            d = _session.get(TMDB_DETAIL_URL.format(id=tid),
                             params={**auth_params, "language": "zh-CN"},
                             headers=auth_headers, timeout=10)
            if d.ok:
                dj = d.json()
                info["overview"] = dj.get("overview", "") or ""
                info["backdrop"] = dj.get("backdrop_path")
                info["release_date"] = dj.get("release_date", "") or info["release_date"]
                info["vote_average"] = dj.get("vote_average")
        except requests.RequestException:
            pass
    return info


def _extract_detail_from_web(tmdb_id, timeout=12):
    """请求 TMDB 网页详情页，提取中文简介 / backdrop / 海报 / 评分。"""
    url = WEB_DETAIL_URL.format(id=tmdb_id)
    r = _web_session.get(url, timeout=timeout, proxies=DIRECT, headers=ZH_HEADERS)
    if not r.ok:
        return None
    h = r.text

    # 中文简介：优先 meta description（跟随 Accept-Language 返回中文）
    overview = ""
    m = re.search(r'<meta\s+name="description"\s+content="(.*?)"\s*/?>', h, re.S)
    if m:
        overview = html.unescape(m.group(1)).strip()
    # 兜底：正文里的 <div class="overview"> ... <p>
    if not overview:
        m = re.search(r'<div class="overview"[^>]*>\s*<p>(.*?)</p>', h, re.S)
        if m:
            overview = re.sub(r"<[^>]+>", "", html.unescape(m.group(1))).strip()
    # 过滤掉 TMDB 默认英文站点描述（无简介时的占位文案）
    if overview.lower().startswith("the movie database"):
        overview = ""

    # 氛围大图：优先横版 backdrop（w533_and_h300_face / w780 / original），
    # 避免误用竖版海报当背景。TMDB 详情页的 <img class="backdrop ..."> 即横版剧照。
    backdrop = None
    bm = re.search(r'<img[^>]*class="[^"]*backdrop[^"]*"[^>]*src="([^"]+)"', h, re.S)
    if not bm:
        bm = re.search(r'<img[^>]*src="([^"]+)"[^>]*class="[^"]*backdrop[^"]*"', h, re.S)
    if bm:
        mm = re.search(r'/t/p/[^/]+/([a-zA-Z0-9]+\.(?:jpg|webp|png))', html.unescape(bm.group(1)))
        if mm:
            backdrop = mm.group(1)
    if not backdrop:
        # 兜底：匹配横版尺寸路径
        bm = re.search(r'/t/p/(?:w533_and_h300_face|w780|w1280)/([a-zA-Z0-9]+\.(?:jpg|webp|png))', h)
        if bm:
            backdrop = bm.group(1)

    # 海报路径：og:image 通常是竖版海报
    poster = None
    bg = re.search(r'<meta\s+property="og:image"\s+content="(.*?)"\s*/?>', h, re.S)
    if bg:
        mm = re.search(r'/t/p/[^/]+/([a-zA-Z0-9]+\.(?:jpg|webp|png))', html.unescape(bg.group(1)))
        if mm:
            poster = mm.group(1)
    if not poster:
        pm = re.search(r'/t/p/w(?:185|342|500)/([a-zA-Z0-9]+\.(?:jpg|webp|png))', h)
        if pm:
            poster = pm.group(1)

    # TMDB 用户评分：<div class="user_score_chart" data-percent="87"> → 8.7
    vote = None
    vm = re.search(r'user_score_chart[^>]*data-percent="(\d+)"', h)
    if vm:
        try:
            vote = round(float(vm.group(1)) / 10.0, 1)
        except ValueError:
            vote = None

    return {
        "overview": overview,
        "backdrop": backdrop,
        "poster": poster,
        "release_date": "",
        "vote_average": vote,
    }


def _fetch_via_web(title, retries=3):
    """网页版通道：搜索页拿 tmdb_id + 海报，再详情页拿中文简介。

    返回 info dict 或 None。带重试，应对并发下的偶发超时/限流。
    """
    year = extract_year(title)
    params = {"query": clean_movie_title(title)}
    if year:
        params["year"] = year

    last_err = None
    for attempt in range(retries):
        try:
            resp = _web_session.get(WEB_SEARCH_URL, params=params,
                                    timeout=15, proxies=DIRECT, headers=ZH_HEADERS)
            if not resp.ok:
                last_err = RuntimeError(f"web search HTTP {resp.status_code}")
                time.sleep(0.6 * (attempt + 1))
                continue

            # 解析搜索结果卡片：取第一个带海报的结果
            tmdb_id, poster_path = None, None
            for m in re.finditer(r'<a[^>]+href="(/movie/(\d+)-[^"]+)"[^>]*>(.*?)</a>',
                                 resp.text, re.S):
                body = m.group(3)
                imgs = re.findall(r'/t/p/[^/]+/([a-zA-Z0-9]+\.(?:jpg|webp|png))', body)
                if imgs:
                    tmdb_id, poster_path = m.group(2), imgs[0]
                    break
            if not tmdb_id:
                return None  # 确实无结果

            info = {
                "poster": poster_path,
                "title": clean_movie_title(title),
                "overview": "",
                "backdrop": None,
                "release_date": "",
                "vote_average": None,
            }
            # 详情页补齐简介与氛围图（内联重试，应对并发下的偶发限流/超时）
            detail = None
            for d_attempt in range(2):
                try:
                    detail = _extract_detail_from_web(tmdb_id)
                    if detail and detail.get("overview"):
                        break  # 拿到简介即成功
                except requests.RequestException:
                    detail = None
                time.sleep(0.4 * (d_attempt + 1))
            if detail:
                info.update({k: v for k, v in detail.items() if v})
                if detail.get("poster"):
                    info["poster"] = detail["poster"]
            return info
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    raise last_err if last_err else RuntimeError("web unknown error")


def _fetch_one(movie_id, title):
    with _cache_lock:
        info_rec = _info_cache.get(str(movie_id))
        if _FILL_MISSING:
            # 补漏模式：只对"已有缓存但简介为空"的影片重抓
            if info_rec and info_rec.get("overview"):
                return "skip", None
        elif str(movie_id) in _path_cache and info_rec:
            # 常规模式：海报+详情都齐全才跳过
            return "skip", None

    try:
        info = _fetch_via_api(title) if _CHANNEL == "api" else _fetch_via_web(title)
        if not info:
            return "none", None
        return "ok", info
    except Exception as e:
        with _print_lock:
            if len(_fail_samples) < 5:
                _fail_samples.append(f"{title[:35]} → {type(e).__name__}: {str(e)[:60]}")
        return "fail", None


# ---------- 主流程 ----------
def main():
    global _CHANNEL, _FILL_MISSING
    parser = argparse.ArgumentParser(description="批量拉取电影海报 + 中文简介（API/网页版自动切换）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 部（0=全部）")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数（网页版建议≤6）")
    parser.add_argument("--channel", choices=["auto", "api", "web"], default="auto",
                        help="通道选择：auto自动/api官方/web网页版(免加速)")
    parser.add_argument("--fill-missing", action="store_true",
                        help="补漏模式：只重抓简介为空的影片（全量跑完后用）")
    args = parser.parse_args()
    _FILL_MISSING = args.fill_missing
    if _FILL_MISSING:
        print("→ 补漏模式：仅重抓简介为空的影片\n")

    if args.channel == "auto":
        print("正在检测 TMDB 网络通道...")
        api_ok, api_reason = probe_api()
        web_ok, web_reason = probe_web()
        print(f"  官方 API : {'可用' if api_ok else '不可用'} ({api_reason})")
        print(f"  网页版   : {'可用' if web_ok else '不可用'} ({web_reason})")
        if api_ok:
            _CHANNEL = "api"
            print("→ 使用官方 API 通道（快速、中文信息准确）\n")
        elif web_ok:
            _CHANNEL = "web"
            print("→ 官方 API 被阻断，自动切换网页版通道（无需开加速器）\n")
        else:
            print("\n错误：API 和网页版都不可用。请检查网络，或开启加速器后重试。")
            sys.exit(1)
    else:
        _CHANNEL = args.channel
        print(f"→ 使用 {_CHANNEL.upper()} 通道（强制）\n")

    movies = pd.read_csv(os.path.join(BASE_DIR, "data", "movies.csv"))
    ratings = pd.read_csv(os.path.join(BASE_DIR, "data", "ratings.csv"))
    cnt = ratings.groupby("movieId").size().rename("cnt")
    movies = movies.merge(cnt, on="movieId", how="left").fillna({"cnt": 0})
    movies = movies.sort_values(["cnt", "movieId"], ascending=False)
    if args.limit:
        movies = movies.head(args.limit)

    total = len(movies)
    done = 0
    print(f"共 {total} 部电影待处理，海报缓存 {len(_path_cache)} 条，详情缓存 {len(_info_cache)} 条，并发 {args.workers}\n")

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch_one, int(row.movieId), row.title): row
            for row in movies.itertuples()
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result, info = future.result()
            except Exception:
                result, info = "fail", None
            _stats[result] = _stats.get(result, 0) + 1
            done += 1

            if info:
                with _cache_lock:
                    if info.get("poster"):
                        _path_cache[str(row.movieId)] = info["poster"]
                    _info_cache[str(row.movieId)] = {
                        "overview": info.get("overview", ""),
                        "backdrop": info.get("backdrop"),
                        "release_date": info.get("release_date", ""),
                        "vote_average": info.get("vote_average"),
                        "poster": info.get("poster"),
                        "title": info.get("title", ""),
                    }
                    if info.get("overview"):
                        _stats["info"] = _stats.get("info", 0) + 1

            if done % 50 == 0 or done == total:
                with _cache_lock:
                    save_cache_if_needed(_path_cache, PATH_CACHE_FILE)
                    save_cache_if_needed(_info_cache, INFO_CACHE_FILE)
            if done % 100 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(f"进度 {done}/{total}  成功 {_stats['ok']}  有简介 {_stats['info']}  "
                      f"无海报 {_stats['none']}  失败 {_stats['fail']}  跳过 {_stats['skip']}  "
                      f"用时 {elapsed:.0f}s  预计剩余 {eta/60:.1f}min", flush=True)

    with _cache_lock:
        save_cache_if_needed(_path_cache, PATH_CACHE_FILE)
        save_cache_if_needed(_info_cache, INFO_CACHE_FILE)
    elapsed = time.time() - start
    print(f"\n完成！成功 {_stats['ok']}，其中带中文简介 {_stats['info']}，"
          f"无海报 {_stats['none']}，失败 {_stats['fail']}，跳过 {_stats['skip']}，"
          f"总详情缓存 {len(_info_cache)} 条，用时 {elapsed/60:.1f} 分钟")
    print(f"海报缓存：{PATH_CACHE_FILE}")
    print(f"详情缓存：{INFO_CACHE_FILE}")

    if _fail_samples:
        print("\n失败样本（前5条，供诊断）：")
        for s in _fail_samples:
            print("  " + s)


if __name__ == "__main__":
    main()
