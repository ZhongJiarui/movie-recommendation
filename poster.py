"""本地海报生成器。

当没有配置 TMDB API key，或 TMDB 请求失败 / 该影片没有海报时，
根据片名与类型即时生成一张美观的 SVG 渐变海报（data URI），
让缺海报的影片也有一张统一的占位图。

纯标准库实现，无第三方依赖，输出可直接作为 <img src> 使用。
"""

import base64
import hashlib
import html
import re

from functools import lru_cache


# 深色影院风渐变配色（每张海报按片名稳定地取其中一组）
GRADIENTS = [
    ("#2b1055", "#6a4fc2", "#c9a7ff"),
    ("#101820", "#243b55", "#5c7cfa"),
    ("#1a0b2e", "#7b2ff7", "#f107a3"),
    ("#0f2027", "#2c5364", "#66a6ff"),
    ("#2b0d1f", "#8e2de2", "#ff7e5f"),
    ("#141e30", "#355c7d", "#c06c84"),
    ("#1d1b2e", "#4a2c7a", "#f0a868"),
    ("#0b101a", "#2c3e50", "#e67e22"),
]

# 类型 -> 视觉符号（emoji），取影片的主类型
GENRE_ICONS = {
    "Action": "💥", "Adventure": "🧭", "Animation": "🎨",
    "Children": "🧸", "Comedy": "😄", "Crime": "🕵️",
    "Documentary": "🎥", "Drama": "🎭", "Fantasy": "🐉",
    "Film-Noir": "🌑", "Horror": "👻", "Musical": "🎵",
    "Mystery": "🔮", "Romance": "💜", "Sci-Fi": "🚀",
    "Thriller": "🗡️", "War": "⚔️", "Western": "🤠",
}

# 类型 -> 中文标签
GENRE_LABELS = {
    "Action": "动作", "Adventure": "冒险", "Animation": "动画",
    "Children": "家庭", "Comedy": "喜剧", "Crime": "犯罪",
    "Documentary": "纪录", "Drama": "剧情", "Fantasy": "奇幻",
    "Film-Noir": "黑色电影", "Horror": "恐怖", "Musical": "歌舞",
    "Mystery": "悬疑", "Romance": "爱情", "Sci-Fi": "科幻",
    "Thriller": "惊悚", "War": "战争", "Western": "西部",
}


def split_title_year(title):
    """把 'Toy Story (1995)' 拆成 ('Toy Story', '1995')。"""
    title = (title or "").strip()
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return title[:match.start()].strip(), match.group(1)
    return title, ""


def primary_genre(genres):
    """返回主类型英文标识。"""
    if not genres:
        return ""
    first = str(genres).split("|")[0].strip()
    return first


def _stable_int(text):
    digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _wrap_text(text, max_chars=15, max_lines=3):
    """按词（英文）/字（中文）简单折行。"""
    words = (text or "").split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
            if len(current) > max_chars:  # 超长单词强行切断
                chunks = [current[i:i + max_chars]
                          for i in range(0, len(current), max_chars)]
                lines.extend(chunks[:-1])
                current = chunks[-1]
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > sum(map(len, lines)):
        lines[-1] = lines[-1][:max_chars - 1] + "…"
    return lines[:max_lines]


@lru_cache(maxsize=2048)
def generate_poster_data_uri(title, genres=""):
    """生成海报 data URI。相同片名/类型只生成一次。"""
    clean, year = split_title_year(title)
    genre = primary_genre(genres)
    icon = GENRE_ICONS.get(genre, "🎬")
    label = GENRE_LABELS.get(genre, "电影")
    g1, g2, g3 = GRADIENTS[_stable_int(title) % len(GRADIENTS)]

    lines = _wrap_text(clean or "MovieRec")
    title_svg = ""
    base_y = 560
    line_height = 46
    for i, line in enumerate(lines):
        title_svg += (
            f'<tspan x="250" y="{base_y + i * line_height}" '
            f'text-anchor="middle">{html.escape(line)}</tspan>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" viewBox="0 0 500 750">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{g1}"/>
      <stop offset="55%" stop-color="{g2}"/>
      <stop offset="100%" stop-color="{g3}"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="38%" r="55%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="500" height="750" fill="url(#bg)"/>
  <rect width="500" height="750" fill="url(#glow)"/>
  <circle cx="250" cy="300" r="180" fill="#ffffff" opacity="0.06"/>
  <circle cx="250" cy="300" r="130" fill="none" stroke="#ffffff" stroke-opacity="0.18" stroke-width="2"/>
  <circle cx="250" cy="300" r="220" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="2"/>
  <text x="250" y="330" text-anchor="middle" font-size="150">{icon}</text>
  <rect x="0" y="470" width="500" height="280" fill="#0b0b16" opacity="0.32"/>
  <text font-family="Microsoft YaHei,PingFang SC,Arial,sans-serif" font-size="42"
        font-weight="700" fill="#ffffff">{title_svg}</text>
  <line x1="190" y1="{base_y + len(lines) * line_height - 6}" x2="310"
        y2="{base_y + len(lines) * line_height - 6}" stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>
  <text x="250" y="{base_y + len(lines) * line_height + 28}" text-anchor="middle"
        font-family="Microsoft YaHei,PingFang SC,Arial,sans-serif" font-size="24"
        fill="#ffffff" opacity="0.85">{html.escape(label)}{(' · ' + year) if year else ''}</text>
  <text x="250" y="720" text-anchor="middle" font-family="Arial,sans-serif"
        font-size="18" letter-spacing="6" fill="#ffffff" opacity="0.45">MOVIEREC</text>
</svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def get_poster(title, genres="", tmdb_poster=None):
    """优先返回 TMDB 海报，否则返回本地生成的海报。"""
    if tmdb_poster:
        return tmdb_poster
    return generate_poster_data_uri(title, genres)
