# -*- coding: utf-8 -*-
"""拟人化多轮对话引擎。

分层设计：
    ┌──────────────────────────────────────────────┐
    │ 1. 意图识别   本地规则优先，LLM 只在需要时介入 │
    │ 2. 会话记忆   参考片 / 心情 / 类型 / 已推电影   │
    │ 3. 电影检索   永远由 recommender 计算，不编造   │
    │ 4. 回复生成   LLM 润色 → 失败回退到本地模板库   │
    └──────────────────────────────────────────────┘

大模型只负责「听懂人话」和「把话说得自然」，绝不负责给出电影 ID ——
这样即使模型断网、幻觉或额度耗尽，用户拿到的依然是真实存在、有海报的片子。
"""

import random as _random
import re as _re

import llm_client
from recommender import (
    MOODS,
    parse_chat_query,
    recommend_by_chat,
    stats_for,
    _clean_title,
    CHAT_MOOD_KEYWORDS,
    CHAT_GENRE_KEYWORDS,
    CHAT_TITLE_ALIASES,
)

# 心情 / 类型 / 片名词也意味着这是一个实质推荐请求
_SUBSTANTIVE_WORDS = set()
for _words in CHAT_MOOD_KEYWORDS.values():
    _SUBSTANTIVE_WORDS.update(_words)
_SUBSTANTIVE_WORDS.update(CHAT_GENRE_KEYWORDS.keys())
_SUBSTANTIVE_WORDS.update(CHAT_TITLE_ALIASES.keys())


_RNG = _random.Random()

# ---------------------------------------------------------------------------
# 意图定义
# ---------------------------------------------------------------------------
INTENT_GREETING = "greeting"
INTENT_THANKS = "thanks"
INTENT_GOODBYE = "goodbye"
INTENT_IDENTITY = "identity"
INTENT_HELP = "help"
INTENT_RECOMMEND = "movie_request"
INTENT_MORE = "followup_more"
INTENT_CHANGE = "followup_change"
INTENT_LIKE = "feedback_like"
INTENT_DISLIKE = "feedback_dislike"
INTENT_OUT = "out_of_scope"

_VALID_INTENTS = {
    INTENT_GREETING, INTENT_THANKS, INTENT_GOODBYE, INTENT_IDENTITY,
    INTENT_HELP, INTENT_RECOMMEND, INTENT_MORE, INTENT_CHANGE,
    INTENT_LIKE, INTENT_DISLIKE, INTENT_OUT,
}

# 本地关键词表（覆盖绝大多数常见说法）
_GREETING_WORDS = ("你好", "您好", "嗨", "哈喽", "哈罗", "hi", "hello", "hey", "在吗", "在么", "早上好", "晚上好")
_THANKS_WORDS = ("谢谢", "感谢", "谢啦", "多谢", "thx", "thank")
_GOODBYE_WORDS = ("再见", "拜拜", "回见", "晚安", "先走了", "bye", "回头聊")
_IDENTITY_WORDS = ("你是谁", "你叫什么", "你是什么", "你的名字", "who are you", "自我介绍")
_HELP_WORDS = ("怎么用", "能做什么", "你会什么", "帮帮我", "帮助", "使用说明", "what can you do")
_MORE_WORDS = ("再来", "再推荐", "再给", "还要", "更多", "换一批", "换一批吧", "别的呢", "还有呢", "下一批", "继续")
_CHANGE_WORDS = ("换成", "换一个", "换个", "改一下", "换种", "不要这种", "不喜欢这种", "换换", "别要这种", "换了")
_LIKE_WORDS = ("喜欢", "不错", "挺好", "很好", "好棒", "赞", "这个好", "正合我意", "满意", "nice")
_DISLIKE_WORDS = ("不喜欢", "没意思", "看过", "看过了", "换掉", "不要了", "不喜欢这个", "无聊", "不感兴趣", "不喜欢这部")
_OUT_WORDS = ("天气", "几点", "新闻", "股票", "多少钱", "今天日期", "算一下", "翻译", "写代码")

# 通常意味着"这是推荐请求"的信号
_RECOMMEND_SIGNALS = (
    "想看", "推荐", "有没有", "片", "电影", "看什么", "来点", "来几部",
    "想找", "找部", "找几部", "高分", "评分", "类似", "像", "关于",
)


def _has(query, words):
    return any(w in query for w in words)


def classify_intent(query, session=None, use_llm=True):
    """本地规则识别意图；规则无法确定时再请大模型判断。返回 (intent, confidence)。"""
    q = (query or "").strip().lower()
    session = session or {}
    has_history = bool(session.get("last_movies"))

    # ---- 本地规则：高置信度的社交类意图 ----
    if _has(q, _THANKS_WORDS):
        return INTENT_THANKS, 0.95
    if _has(q, _GOODBYE_WORDS) and not _has(q, _RECOMMEND_SIGNALS):
        return INTENT_GOODBYE, 0.9
    if _has(q, _IDENTITY_WORDS):
        return INTENT_IDENTITY, 0.95
    if _has(q, _HELP_WORDS) and not _has(q, _RECOMMEND_SIGNALS):
        return INTENT_HELP, 0.9
    if q in ("你好", "您好", "嗨", "hi", "hello", "哈喽", "在吗", "在么") or (
        _has(q, _GREETING_WORDS) and len(q) <= 8 and not _has(q, _RECOMMEND_SIGNALS)
    ):
        return INTENT_GREETING, 0.95

    # ---- 本地规则：追问与反馈（依赖上下文）----
    if has_history and _has(q, _MORE_WORDS) and not _has(q, ("想看", "有没有", "找部", "找几部")):
        return INTENT_MORE, 0.9
    if has_history and _has(q, _CHANGE_WORDS) and not _has(q, ("像", "类似", "想看")):
        return INTENT_CHANGE, 0.85
    if has_history and _has(q, _DISLIKE_WORDS) and len(q) <= 16:
        return INTENT_DISLIKE, 0.85
    if has_history and _has(q, _LIKE_WORDS) and len(q) <= 12 and not _has(q, _RECOMMEND_SIGNALS):
        return INTENT_LIKE, 0.8

    # ---- 本地规则：推荐信号（含心情/类型/片名）----
    if _has(q, _RECOMMEND_SIGNALS) or _has(q, _SUBSTANTIVE_WORDS) or "《" in query:
        return INTENT_RECOMMEND, 0.85

    # ---- 不确定：交给大模型，失败则温和拉回 ----
    if use_llm and llm_client.is_available():
        data = llm_client.complete_json([
            {
                "role": "system",
                "content": (
                    "你是电影推荐网站的意图分类器。判断用户这句话的意图，只输出 JSON："
                    '{"intent": "greeting|thanks|goodbye|identity|help|movie_request|'
                    'followup_more|followup_change|feedback_like|feedback_dislike|out_of_scope", '
                    '"rewritten": "如果这是推荐或追问请求，把它改写成一句完整的中文推荐需求；否则为空字符串"}。'
                    "不要输出其他任何内容。"
                ),
            },
            {"role": "user", "content": query},
        ], temperature=0.2, max_tokens=200)
        if data and data.get("intent") in _VALID_INTENTS:
            return data["intent"], 0.75

    if _has(q, _OUT_WORDS):
        return INTENT_OUT, 0.6
    # 短句无信号 → 当作打招呼；长句无信号 → 当作模糊推荐请求
    return (INTENT_GREETING if len(q) <= 6 else INTENT_RECOMMEND), 0.4


def _llm_understand(query, session, user_signals):
    """让大模型把自然语言改写成更清晰的推荐需求（可选增强）。返回改写后的句子或 None。"""
    if not llm_client.is_available():
        return None
    ctx = session.get("context") or {}
    parts = []
    if ctx.get("reference_title"):
        parts.append("之前提到参考片《%s》" % ctx["reference_title"])
    if ctx.get("mood"):
        parts.append("偏好氛围：%s" % MOODS.get(ctx["mood"], {}).get("name", ctx["mood"]))
    if ctx.get("genres"):
        parts.append("偏好类型：%s" % "、".join(ctx["genres"][:4]))
    if user_signals.get("top_titles"):
        parts.append("用户高分电影：%s" % "、".join(user_signals["top_titles"][:3]))
    context_text = "；".join(parts) if parts else "无"

    data = llm_client.complete_json([
        {
            "role": "system",
            "content": (
                "你是电影推荐网站的语义理解模块。把用户口语化的需求改写成一句清晰、完整的中文推荐需求。"
                "铁律：以用户这句话为准——用户明确提到的心情、类型、年代、口碑必须完整保留；"
                "对话背景仅供参考，绝不能把背景里的旧片名、旧心情塞进改写结果，"
                "尤其当用户已经提出新的方向时，必须丢弃旧的参考片和氛围。"
                "不要新增用户没提到的具体电影，不要解释。只输出 JSON：{\"rewritten\": \"改写后的句子\"}。"
            ),
        },
        {
            "role": "user",
            "content": "对话背景：%s\n用户这句话：%s" % (context_text, query),
        },
    ], temperature=0.3, max_tokens=200)
    if not data:
        return None
    rewritten = str(data.get("rewritten") or "").strip()
    return rewritten if 2 <= len(rewritten) <= 120 else None


def _llm_reply(query, intent, effective_query, movies, user_signals, session):
    """让大模型生成回复（可选增强）；失败返回 None。"""
    if not llm_client.is_available():
        return None

    movie_names = "、".join(
        "《%s》" % _clean_title(m.get("title", "")) for m in (movies or [])[:4]
    )
    signal_text = ""
    if user_signals.get("top_titles"):
        signal_text = "这位用户喜欢：" + "、".join(user_signals["top_titles"][:3])

    system = (
        "你是 MovieRec 电影推荐网站里亲切又懂片的助手「小影」。"
        "用自然、口语化、有温度的中文回复，1～3 句话、总共不超过 90 字，可以带一个 emoji。"
        "严格遵守：\n"
        "1. 只能提到下面列出的推荐电影，绝不能编造片名或信息；\n"
        "2. 不要输出列表、标题、markdown 或标签；\n"
        "3. 回复要贴合用户当前的这句话和意图，不要每句都是同一套开场白；\n"
        "4. 遇到闲聊就自然接话，遇到推荐就简短说明为什么这几部合适。"
    )
    context = (
        "用户意图：%s\n用户这句话：%s\n实际推荐请求：%s\n推荐电影：%s\n%s"
        % (intent, query, effective_query, movie_names or "（无，纯聊天）", signal_text)
    )
    text = llm_client.complete([
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ], temperature=0.85, max_tokens=220)
    if not text:
        return None
    text = text.strip().strip('"').strip("'")
    # 过滤模型偶尔吐出的 markdown 前缀
    text = _re.sub(r"^(小影[：:]\s*|助手[：:]\s*)", "", text)
    return text if 4 <= len(text) <= 200 else None


# ---------------------------------------------------------------------------
# 本地回复模板
# ---------------------------------------------------------------------------
_T_GREETING = [
    "嗨～来得正好！我刚翻完片库，正愁没人聊电影呢 😄 你现在想看什么感觉的？",
    "你好呀！别急着选类型，先说说今天的心情，或者一部你反复刷的电影，我就能顺着味儿给你找片。",
    "哟，你来啦！片海茫茫，说说你想找什么，我帮你捞 🎬",
    "嘿！我在这儿等着呢～是想找部电影消磨时间，还是纯粹想聊两句？都行～",
]
_T_THANKS = [
    "客气啥～能挑到你爱看的片子，我比你还开心。下次想换口味了随时喊我 😊",
    "别说谢，选片可是我的强项。看得愉快！",
    "嘿嘿，乐意效劳～有好片子记得回来告诉我感觉怎么样。",
]
_T_GOODBYE = [
    "回见！愿你今晚遇到一部让你忘了时间的电影 🎬",
    "拜拜～下次想找片了，我还在老地方等你。",
    "晚安！好梦，也祝你好片～",
]
_T_IDENTITY = [
    "我是 MovieRec 的观影搭子「小影」，专门帮你从 9000 多部片子里挑对味的。"
    "你说大白话就行——「像星际穿越但别那么烧脑」这种我也听得懂。"
    "而且你用得越多，我越摸得准你的口味 😉",
    "我叫小影，是你在这儿专属的选片助手。能聊天、能推荐、还能记住你不爱看的片。"
    "放心，我推荐的每部都是片库里真实存在的，绝不瞎编～",
]
_T_HELP = [
    "用法超简单：直接说「今晚想笑一场」「像盗梦空间那种烧脑片」「90 年代的高分经典」就行。"
    "看完还能跟我说「再来几部」「换个轻松点的」，我会接着调整 🎯",
    "你只要描述想看的感觉，剩下的交给我。也可以先点上面的示例问题找找感觉～",
]
_T_OUT = [
    "哈哈这个超出我的业务范畴啦～不过聊电影我可在行。想看什么类型的？",
    "这个我真帮不上忙，但说到选片我还没输过。来，说说你想看的感觉？",
    "跑题啦～要不咱们聊回电影？你现在想找点什么感觉的 😄",
]
_T_MORE = [
    "没问题，再给你补几部同款。这次特意避开了刚才推过的，放心挑～",
    "好嘞，又翻了一批出来，口味还是照刚才的路子来的。",
    "收到，继续顺着这个感觉找。看看这几部合不合胃口？",
]
_T_LIKE = [
    "好眼光！这部确实值得看。要我顺着这个感觉再给你找几部吗？",
    "哈哈 satisfaction！那我记住这个口味了，下次优先给你找同款 😄",
]
_T_DISLIKE_GENERIC = [
    "收到，我记下了。那换个方向再给你找几部？",
    "好嘞，这类先放一边。说说你想换成什么感觉的？",
]


def _recommend_reply_local(parsed, query, effective_query, movies, user_signals, session):
    """本地模板生成推荐回复：结合上下文、用户口味和实际结果，避免千篇一律。"""
    pieces = []
    turn = int(session.get("turn") or 0)

    # 1) 开场：承接上下文 / 用户口味
    if turn > 0:
        pieces.append(_RNG.choice(["好嘞，按你说的来～", "懂了，这就给你调整。", "收到，换个方向试试。"]))
    elif user_signals.get("top_titles"):
        top = user_signals["top_titles"][0]
        pieces.append("看你之前给《%s》打了高分，我大概摸到你的路子了。" % _clean_title(top))
    else:
        pieces.append(_RNG.choice(["这个有意思，让我翻翻片库～", "没问题，我来想想。", "好嘞，这就给你找。"]))

    # 2) 说明推荐依据（用人话复述意图）
    bits = []
    if parsed.get("reference_title"):
        bits.append("和《%s》一个路子" % _clean_title(parsed["reference_title"]))
    if parsed.get("mood"):
        bits.append(MOODS[parsed["mood"]]["name"])
    genre_cn = parsed.get("genre_cn") or ""
    if genre_cn:
        bits.append(genre_cn.split("、")[0] + "向")
    if parsed.get("decade"):
        bits.append("%s 年代" % parsed["decade"])
    if parsed.get("quality"):
        bits.append("高分口碑")
    if bits:
        pieces.append("按你说的「%s」，我挑了这几部。" % "、".join(bits[:3]))
    elif effective_query and effective_query != query:
        pieces.append("按你说的「%s」，我挑了这几部。" % effective_query[:26])

    # 3) 结尾：点出最推荐的一部 + 邀请反馈
    if movies:
        first = _clean_title(movies[0].get("title", ""))
        rating = movies[0].get("avg_rating")
        tail = _RNG.choice([
            "第一部《%s》我觉得最对味，先看看？" % first,
            "如果只挑一部，我投《%s》一票。" % first,
            "特别想安利《%s》，看完告诉我感觉，我再帮你调。" % first,
        ])
        if rating and rating >= 4.2:
            tail += " 口碑也很稳（%s 分）。" % rating
        pieces.append(tail)
    else:
        pieces.append("这次没找到特别贴合的，换个说法我再试试？")

    return " ".join(pieces)


def _suggestions(intent, has_movies, has_history):
    if intent in (INTENT_GREETING, INTENT_IDENTITY, INTENT_HELP, INTENT_OUT):
        return ["想看像星际穿越那样的科幻片", "今晚想笑一场", "来点烧脑悬疑高分"]
    if intent in (INTENT_THANKS, INTENT_GOODBYE):
        return ["推荐部电影吧", "换个口味", "你好"]
    if has_movies:
        base = ["再来几部类似的", "换个轻松点的", "要更高分的", "换个类型"]
        return base[:3] if not has_history else base
    return ["换个说法试试", "推荐几部高分电影"]


def _update_context(session, parsed, query, movie_ids):
    """把本轮结果写入会话上下文（只保留可 JSON 序列化的基础类型）。"""
    ctx = session.get("context") or {}
    if parsed.get("reference_title"):
        ctx["reference_title"] = parsed["reference_title"]
    if parsed.get("mood"):
        ctx["mood"] = parsed["mood"]
    if parsed.get("genres"):
        ctx["genres"] = list(parsed["genres"].keys())[:5]
    if parsed.get("decade"):
        ctx["decade"] = parsed["decade"]
    if parsed.get("quality"):
        ctx["quality"] = True
    ctx["query"] = query
    session["context"] = ctx
    session["last_movies"] = [
        {"id": int(m["movieId"]), "title": m.get("title", "")}
        for m in (movie_ids or [])
    ]
    session["turn"] = int(session.get("turn") or 0) + 1


def _effective_query(intent, query, session, user_signals, use_llm):
    """把追问/模糊请求还原成完整的推荐需求文本。"""
    ctx = session.get("context") or {}
    if intent == INTENT_MORE:
        # 追问：以上一轮需求为基础；若本轮带了新条件（如"再来几部悬疑的"）则合并进去
        base = ctx.get("query") or ""
        if base and (_has(query, _SUBSTANTIVE_WORDS) or "《" in query):
            return (base + "，" + query).strip("，"), "followup"
        return base or query, "followup"
    if intent == INTENT_CHANGE:
        # 把新指令贴在旧需求后，让解析器同时看到"换"和具体方向
        return (ctx.get("query", "") + "，" + query).strip("，"), "changed"
    if intent in (INTENT_LIKE, INTENT_DISLIKE, INTENT_GREETING, INTENT_THANKS,
                  INTENT_GOODBYE, INTENT_IDENTITY, INTENT_HELP, INTENT_OUT):
        return query, "chitchat"

    # 推荐请求：
    # 若当前这句话已自带明确的推荐信号（心情/类型/片名/年代/口碑），直接用原句 ——
    # 本地解析器对这些结构化信号处理最准，让 LLM 改写反而容易把历史上下文混进来
    # （例如上一轮推了动画片，本轮说"烧脑悬疑"，改写会被污染成"像 Toy Story 的搞笑悬疑"）。
    # 只有口语化、无明确信号的模糊需求，才交给 LLM 结合上下文改写。
    if use_llm and not _has(query, _SUBSTANTIVE_WORDS) and "《" not in query:
        rewritten = _llm_understand(query, session, user_signals)
        if rewritten:
            return rewritten, "llm"
    return query, "local"


def handle_chat(query, session, user_signals=None, use_llm=True):
    """处理一轮对话。

    session: dict，会被就地更新并写回 Flask session
    user_signals: {"ratings": {}, "favorites": [], "disliked": [], "top_titles": []}
    返回可直接 jsonify 的 dict。
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "说点什么吧，比如「想看烧脑的科幻片」"}
    if len(query) > 150:
        query = query[:150]

    user_signals = user_signals or {}
    session.setdefault("context", {})
    session.setdefault("last_movies", [])
    session.setdefault("turn", 0)
    session.setdefault("avoid", [])

    # 1) 意图识别
    intent, confidence = classify_intent(query, session, use_llm=use_llm)

    # 2) 还原完整需求
    effective_query, source = _effective_query(intent, query, session, user_signals, use_llm)

    # 3) 召回电影（闲聊类不触发检索）
    movies = []
    parsed = {}
    if intent in (INTENT_RECOMMEND, INTENT_MORE, INTENT_CHANGE):
        # 已明确不喜欢的电影 + 用户黑名单，一并排除
        avoid = set(int(x) for x in session.get("avoid", []))
        avoid |= set(int(x) for x in (user_signals.get("disliked") or []))
        # 追问时额外排除上一轮已推荐的，保证"再来几部"是真的新片
        if intent == INTENT_MORE:
            avoid |= set(int(m["id"]) for m in session.get("last_movies", []))

        try:
            parsed, movies = recommend_by_chat(
                effective_query, 8,
                user_ratings=user_signals.get("ratings") or {},
                disliked_ids=list(avoid),
                favorites=user_signals.get("favorites") or [],
            )
        except Exception:
            parsed, movies = {}, []

    # 4) 生成回复：LLM 润色优先，失败回退本地模板
    reply = None
    reply_source = "local"
    if use_llm and llm_client.is_available():
        reply = _llm_reply(query, intent, effective_query, movies, user_signals, session)
        if reply:
            reply_source = "llm"
    if not reply:
        reply = _local_reply(intent, query, effective_query, movies, user_signals, session, parsed)

    # 5) 更新会话状态
    if movies:
        _update_context(session, parsed, query, movies)
    else:
        session["turn"] = int(session.get("turn") or 0) + 1

    # 6) 组装返回值（保持对旧前端字段兼容）
    summary = parsed.get("summary") or ""
    if not summary and intent in (INTENT_RECOMMEND, INTENT_MORE, INTENT_CHANGE):
        summary = "正在为你寻找合适的电影"
    tags = parsed.get("tags") or []

    return {
        "ok": True,
        "reply": reply,
        "summary": summary,
        "tags": tags,
        "movies": movies,  # 原始电影 dict，由 app 层统一 _prepare 补海报后序列化
        "suggestions": _suggestions(intent, bool(movies), bool(session.get("last_movies"))),
        "intent": intent,
        "mode": reply_source,
    }


def _local_reply(intent, query, effective_query, movies, user_signals, session, parsed):
    if intent == INTENT_GREETING:
        return _RNG.choice(_T_GREETING)
    if intent == INTENT_THANKS:
        return _RNG.choice(_T_THANKS)
    if intent == INTENT_GOODBYE:
        return _RNG.choice(_T_GOODBYE)
    if intent == INTENT_IDENTITY:
        return _RNG.choice(_T_IDENTITY)
    if intent == INTENT_HELP:
        return _RNG.choice(_T_HELP)
    if intent == INTENT_OUT:
        return _RNG.choice(_T_OUT)
    if intent == INTENT_MORE:
        return _RNG.choice(_T_MORE) + (" 这几部先看看？" if movies else "")
    if intent == INTENT_LIKE:
        return _RNG.choice(_T_LIKE)
    if intent == INTENT_DISLIKE:
        # 尽量识别用户指的是哪部，记入黑名单
        last = session.get("last_movies") or []
        hit = None
        for m in last:
            title = _clean_title(m.get("title", ""))
            if title and (title in query or query in title):
                hit = m
                break
        if hit:
            avoid = list(session.get("avoid") or [])
            if int(hit["id"]) not in avoid:
                avoid.append(int(hit["id"]))
            session["avoid"] = avoid  # 重新赋值，确保 Flask session 检测到变更
            return "明白，《%s》我记下了，以后不再推它。那换几部别的给你？" % _clean_title(hit["title"])
        return _RNG.choice(_T_DISLIKE_GENERIC)
    return _recommend_reply_local(parsed, query, effective_query, movies, user_signals, session)


def reset_session(session):
    """清空对话上下文。"""
    for key in ("context", "last_movies", "avoid", "turn"):
        session.pop(key, None)


def ai_status():
    """供页面展示当前 AI 模式。"""
    return llm_client.status()
