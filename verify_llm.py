# -*- coding: utf-8 -*-
"""检查大模型配置是否接通。

用法：填好 .env 里的 LLM 配置后，在项目目录运行：
    .venv/Scripts/python.exe verify_llm.py

脚本会依次检查：配置是否读取 → 网络是否可达 → 模型是否返回内容与电影解析结果。
"""

import sys
import time

import config
import llm_client


def line(title=""):
    print("\n" + "=" * 58)
    if title:
        print("  " + title)
        print("=" * 58)


def main():
    line("AI 助手 · 大模型接通验证")

    # 1) 配置检查
    print("\n[1/3] 检查配置")
    print("  服务商 LLM_PROVIDER : %s" % (config.LLM_PROVIDER or "（未设置）"))
    print("  密钥   LLM_API_KEY   : %s" % (
        (config.LLM_API_KEY[:6] + "..." + config.LLM_API_KEY[-4:])
        if config.LLM_API_KEY else "（未设置）"
    ))
    print("  模型   LLM_MODEL     : %s" % (config.LLM_MODEL or "（用服务商默认值）"))
    print("  接口   LLM_BASE_URL  : %s" % (config.LLM_BASE_URL or "（用服务商默认值）"))

    if not config.LLM_API_KEY:
        print("\n  ✗ 还没有配置 API Key。")
        print("  请复制 .env.llm.example 的配置到 .env，填入 LLM_API_KEY 后重试。")
        return 1
    if not llm_client.is_available():
        print("\n  ✗ 配置不完整，模型不可用。")
        return 1
    print("  ✓ 配置已读取")

    st = llm_client.status()
    preset_model = st.get("model") or "（未知）"
    print("  将使用的模型：%s" % preset_model)

    # 2) 实际调用
    print("\n[2/3] 向 %s 发送一条测试消息…" % (config.LLM_PROVIDER or "custom"))
    t0 = time.time()
    reply = llm_client.complete(
        [
            {"role": "system", "content": "你是电影推荐助手。用一句中文简短回复。"},
            {"role": "user", "content": "你好，介绍一下你自己"},
        ],
        temperature=0.7,
        max_tokens=100,
        timeout=config.LLM_TIMEOUT,
    )
    elapsed = time.time() - t0

    if not reply:
        print("  ✗ 没有收到回复。可能原因：")
        print("    - 网络无法访问该服务商域名（国内一般可直连智谱/DeepSeek）")
        print("    - API Key 错误、过期或未实名（智谱需实名认证）")
        print("    - 模型 id 不对，可在 .env 用 LLM_MODEL 覆盖")
        print("    - 免费额度触发限流（429），稍后再试")
        print("  提示：可在 app.py 启动日志里看具体报错。")
        return 1
    print("  ✓ 收到回复（耗时 %.1fs）：" % elapsed)
    print('      "%s"' % reply.replace("\n", " "))

    # 3) 验证 JSON 解析（对话引擎靠它理解意图）
    print("\n[3/3] 验证语义理解（JSON 解析）…")
    data = llm_client.complete_json(
        [
            {"role": "system", "content":
                "你是电影推荐网站的意图分类器。只输出 JSON："
                '{"intent": "movie_request|greeting|out_of_scope", "rewritten": "改写成完整推荐需求"}'},
            {"role": "user", "content": "想看像星际穿越那样的科幻片"},
        ],
        temperature=0.2,
        max_tokens=120,
    )
    if data and data.get("intent"):
        print("  ✓ 语义理解正常：intent=%s，改写=%s" % (
            data["intent"], data.get("rewritten", "")))
    else:
        print("  ! JSON 解析未通过，但纯文本回复正常，助手仍可用（会用本地意图识别兜底）")

    line("验证完成")
    print("  接通成功！重启 Flask 服务后，/chat 页面顶部会显示「大模型增强」。")
    print("  重启命令：先停掉旧服务，再运行 .venv\\Scripts\\python.exe app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
