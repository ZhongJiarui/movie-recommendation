"""项目配置：从 .env 加载密钥，避免硬编码在源码中。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def _load_env(path):
    """极简 .env 解析器，无需第三方依赖。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            # 环境变量优先级更高
            os.environ.setdefault(key, value)


_load_env(ENV_PATH)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BEARER_TOKEN = os.environ.get("TMDB_BEARER_TOKEN", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "movierec-dev-secret-key")

# ---------------------------------------------------------------------------
# 可选大模型配置（用于对话助手理解提问和生成回复）
# 支持 provider: deepseek / zhipu / qwen / moonshot / openai / custom
# 不配置时用本地引擎，不联网也能跑。
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
# 单次请求超时（秒），避免网络问题阻塞页面
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "6") or 6)
