# Translation-Assistant
A small-scale AI application backend interface that connects to large language models, providing functions such as Chinese to English translation, English to Chinese translation, and text summarization.

项目结构概览

- 根目录
  - `.env` ：环境变量（例如模型、API Key、Base URL 等），不应提交密钥到仓库
  - `requirements.txt` ：Python 依赖
  - `README.md`
  - `main.py` ：应用启动、依赖注入与 TaskManager（异步任务管理）
- 后端应用
  - `app`
    - `config.py` ：配置管理（模型、超时、Base URL 等）
    - `api`
      - `routes.py` ：HTTP 路由定义（流式与异步接口、功能列表、任务列表等）
    - `clients`
      - `qwen_client.py` ：上游模型客户端（支持 SDK 不可用时的 HTTP 兼容模式）
    - `services`
      - `translation.py` ：翻译服务（双向，支持流式）
      - `summarization.py` ：摘要服务（可控要点数、目标语言，支持流式）
      - `chat.py` ：聊天服务（流式）
- 前端测试页面
  - `tests`
    - `index.html` ：主测试页（功能列表展示与流式接口测试）
    - `async.html` ：异步任务测试页（提交、轮询、任务列表总览与结果呈现）
接口清单

- 功能列表
  - GET /api/functions
    - 返回后端已实现的功能清单，包含流式与异步任务接口及调用方式（便于前端动态展示与自描述）
- 异步任务
  - POST /api/tasks/translate
    - 请求体：{ text: string, direction: "zh_to_en" | "en_to_zh" }
    - 返回：{ task_id: string }
  - POST /api/tasks/summarize
    - 请求体：{ text: string, target_lang?: string, max_points?: number }
    - 返回：{ task_id: string }
  - GET /api/tasks/status?task_id=xxx
    - 返回：{ id: string, status: "pending" | "running" | "succeed" | "failed", type: string, result?: string, error?: string }
    - 说明：当 status 为 succeed 时，result 为具体结果文本；failed 时查看 error
  - GET /api/tasks/list
    - 返回：{ tasks: Array<{ id: string, status: string, type: string, has_result: boolean, has_error: boolean }> }
    - 说明：为列表总览设计的简要视图，不包含 params 与完整 result
- 流式（SSE）接口
  - 提供翻译、摘要、聊天的流式输出，服务层以生成器返回文本片段，路由层封装为 SSE 推送
  - 具体路径以 /api/functions 输出为准（例如翻译 zh_to_en_stream / en_to_zh_stream、摘要 summarize_stream、聊天 chat_stream）
功能说明

- 翻译
  - 中文 → 英文、英文 → 中文，两种方向
  - 支持流式输出（SSE）和异步任务（提交后轮询结果）
  - 参考： `translation.py`
- 摘要
  - 对长文本进行要点式或短段落的摘要
  - 可指定目标语言与要点数（默认 5）
  - 支持流式输出和异步任务
  - 参考： `summarization.py`
- 聊天
  - 针对通用对话的流式输出
  - 参考： `chat.py`
- 异步任务管理
  - 内存版 TaskManager 管理任务生命周期（pending → running → succeed/failed）
  - 提供任务列表总览与状态查询接口，便于前端表格展示与自动刷新
  - 参考： `main.py`
- 前端测试与呈现
  - 主测试页 index.html：包含“功能列表”卡片，可调用 GET /api/functions 并展示；支持流式接口体验
  - 异步测试页 async.html：
    - 支持翻译与摘要的提交，对返回的 task_id 自动写入输入框
    - 支持轮询任务状态（含 JSON 原始显示与“漂亮展示区”，摘要按行转要点列表，翻译按等宽预格式文本展示）
    - 任务列表总览表格：自动刷新、状态徽章、每行操作（复制ID、详情、轮询），并附带原始 JSON 显示
运行与验证

- 启动后端：python -m uvicorn main:app --host 0.0.0.0 --port 8002
- 访问测试页
  - 异步页： http://localhost:8002/tests/async.html
  - 主测试页： http://localhost:8002/tests/index.html


.env 配置
QWEN_API_KEY=yours api-key
QWEN_MODEL=qwen-turbo
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
Timeout=30              # 接口超时时间（秒）