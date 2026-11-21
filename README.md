# 光影荐影

一个用 MovieLens 数据做的电影推荐网站。后端 Flask，推荐部分混合了协同过滤、内容相似和热度打分；对话助手接大模型理解提问，海报和简介来自 TMDB。

## 功能

- **个性化推荐**：SVD 隐语义 + TF-IDF 类型相似 + 贝叶斯质量分 + 热度，再用 MMR 压掉同质化。每条结果带推荐理由和匹配度
- **对话找片**：直接说「今晚想看烧脑的科幻片」，助手会多轮追问再给片单。没配大模型就走本地引擎
- **心情推荐**：按「想笑一场」「深夜独处」这类场景找片
- **探索盲盒**：从冷门但评分不错的片子里随机挑
- **口味 DNA**：根据评分算出六个类型的偏好，画成雷达图，也能看随时间的变化
- **年度报告**：评分分布、偏好类型、观影关键词
- **社交推荐**：找口味相近的人，看他们打高分的片子
- **用户功能**：注册登录、评分、收藏、不感兴趣反馈、登录失败锁定
- **海报与简介**：TMDB 数据抓到本地缓存，页面读缓存不卡

## 技术栈

Python + Flask，算法用 pandas / numpy / scikit-learn（TruncatedSVD、TF-IDF），数据存 SQLite，前端 Jinja2 + 原生 CSS/JS。外部依赖 TMDB 和可选的大模型。

## 本地运行

```bash
pip install -r requirements.txt
python init_db.py
python app.py
```

打开 http://127.0.0.1:5000。

## 配置（可选）

复制示例文件：

```bash
cp .env.llm.example .env
```

| 配置项 | 作用 |
|---|---|
| `TMDB_API_KEY` | 抓新片海报和简介。不填就用仓库里已有的缓存 |
| `TMDB_BEARER_TOKEN` | TMDB Bearer Token，可不填 |
| `SECRET_KEY` | Flask 会话签名，部署时换成随机字符串 |
| `LLM_PROVIDER` / `LLM_API_KEY` | 对话助手接大模型，支持 zhipu / deepseek / qwen / moonshot / openai |
| `LLM_MODEL` | 模型名，留空用服务商默认免费模型 |

什么都不填也能跑：海报简介走缓存，助手走本地引擎。

## 文件说明

```
app.py              Flask 路由
recommender.py      推荐引擎
chat_engine.py      对话意图与多轮上下文
llm_client.py       大模型调用，带降级
tmdb.py             TMDB 抓取与缓存
poster.py           海报路径处理
database.py         数据库与表结构
config.py           配置读取
data/               MovieLens 数据
static/ templates/  样式和页面
poster_paths.json   海报缓存
movie_info.json     简介缓存
```

## 数据

MovieLens 100k：约 9700 部电影、10 万条评分，文件在 `data/` 下。
