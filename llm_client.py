"""可选大模型接入层。

设计原则：
1. 只做「理解语言」和「生成回复文案」，不参与电影检索 —— 电影永远来自本地推荐引擎，
   避免大模型编造不存在的片名。
2. 任何网络/配置/额度问题都静默降级为 None，由上层 chat_engine 回退到本地引擎。
3. 仅用标准库 urllib，不引入新依赖；支持所有 OpenAI chat/completions 兼容服务。
"""

import json
import urllib.request
import urllib.error

import config


# 常见服务商的 OpenAI 兼容端点预设（用户只需填 API Key 即可）
_PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        # glm-4-flash 是智谱长期稳定的免费模型（实测可用、不过期）。
        # 新版 glm-4.7-flash 常因访问量过大返回 429，故不作为默认。
        # 需要用其他模型时可在 .env 用 LLM_MODEL 覆盖，如 glm-4-air / glm-4-flashx。
        "model": "glm-4-flash",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


def is_available():
    """是否配置了可用的大模型。"""
    return bool(config.LLM_API_KEY)


def status():
    """返回配置状态（不含密钥），供前端展示当前 AI 模式。"""
    if not config.LLM_API_KEY:
        return {
            "enabled": False,
            "provider": "",
            "model": "",
            "reason": "未配置 API Key，使用本地智能引擎",
        }
    preset = _PROVIDERS.get(config.LLM_PROVIDER, {})
    return {
        "enabled": True,
        "provider": config.LLM_PROVIDER or "custom",
        "model": config.LLM_MODEL or preset.get("model", "自定义模型"),
        "base_url": config.LLM_BASE_URL or preset.get("base_url", ""),
        "reason": "",
    }


def _endpoint():
    preset = _PROVIDERS.get(config.LLM_PROVIDER, {})
    base = config.LLM_BASE_URL or preset.get("base_url", "")
    model = config.LLM_MODEL or preset.get("model", "")
    if not base or not model:
        return "", ""
    return base.rstrip("/") + "/chat/completions", model


def complete(messages, temperature=0.7, max_tokens=700, timeout=None):
    """调用大模型返回文本；失败返回 None（由上层降级）。"""
    if not is_available():
        return None
    url, model = _endpoint()
    if not url or not model:
        return None

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.LLM_API_KEY,
            "User-Agent": "MovieRec-AI-Assistant/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout or config.LLM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _log("大模型调用失败，已降级到本地引擎: %s" % exc)
        return None

    try:
        choices = body.get("choices") or []
        if not choices:
            return None
        content = (choices[0].get("message") or {}).get("content")
        # 兼容返回内容片段列表的服务商
        if isinstance(content, list):
            content = "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        return str(content).strip() if content else None
    except Exception as exc:
        _log("解析大模型响应失败: %s" % exc)
        return None


def complete_json(messages, **kwargs):
    """调用大模型并容错解析 JSON，失败返回 None。"""
    text = complete(messages, **kwargs)
    if not text:
        return None
    text = text.strip()
    # 去掉 markdown 代码围栏
    if text.startswith("```"):
        text = text.lstrip("`")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower() in ("json", ""):
                text = rest
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        _log("大模型返回的 JSON 无法解析")
        return None


def _log(message):
    """尽量避免模块层依赖 Flask，无应用上下文时静默。"""
    try:
        from flask import current_app
        current_app.logger.warning(message)
    except Exception:
        pass
