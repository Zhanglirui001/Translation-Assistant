from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import load_settings, QwenSettings
from app.clients.qwen_client import QwenClient
from app.services.translation import TranslationService
from app.services.summarization import SummarizationService
from app.services.chat import ChatService
from app.api.routes import router as api_router
from typing import Any, Dict, Callable
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="FastAPI Demo with Config")

# 静态测试页：保持与原有路径一致
app.mount("/tests", StaticFiles(directory="tests", html=True), name="tests")

# CORS 设置：与重构前保持一致
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "X-Requested-With"],
)

# 注册模块化路由
app.include_router(api_router)


# 简单的内存任务管理器：提交后台任务并提供状态轮询
class TaskManager:
    def __init__(self, max_workers: int = 4) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, work_fn: Callable[..., Any], *, task_type: str, params: Dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "status": "pending",
                "type": task_type,
                "params": params,
                "result": None,
                "error": None,
            }
        def _run():
            with self._lock:
                self._tasks[task_id]["status"] = "running"
            try:
                result = work_fn(**params)
                with self._lock:
                    self._tasks[task_id]["status"] = "succeed"
                    self._tasks[task_id]["result"] = result
            except Exception as e:
                with self._lock:
                    self._tasks[task_id]["status"] = "failed"
                    self._tasks[task_id]["error"] = str(e)
        self._executor.submit(_run)
        return task_id

    def get(self, task_id: str) -> Dict[str, Any] | None:
        with self._lock:
            return self._tasks.get(task_id)


@app.on_event("startup")
def on_startup() -> None:
    # 加载配置并挂载到应用状态
    settings: QwenSettings = load_settings()
    app.state.settings = settings

    # 初始化通义千问客户端（自动兼容 HTTP 回退）
    qwen = QwenClient(settings)
    app.state.qwen_client = qwen
    if getattr(qwen, "use_http_fallback", False):
        logger.warning("dashscope SDK 未可用，使用兼容HTTP模式: {}", settings.base_url)

    # 初始化业务服务，并挂载到应用状态供路由复用
    app.state.translation_service = TranslationService(qwen)
    app.state.summarization_service = SummarizationService(qwen)
    app.state.chat_service = ChatService(qwen)
    # 挂载任务管理器（用于异步任务提交与轮询）
    app.state.task_manager = TaskManager(max_workers=4)

    # 避免泄露密钥，仅打印掩码后的信息
    masked_key = (
        settings.api_key[:4] + "..." + settings.api_key[-4:]
        if len(settings.api_key) >= 8
        else "***"
    )
    logger.info(
        "配置已加载: model={}, base_url={}, timeout={}, api_key(masked)={}",
        settings.model,
        settings.base_url,
        settings.timeout,
        masked_key,
    )


# 提供便捷获取配置的方法（与重构前保持功能一致）
def get_settings() -> QwenSettings:
    return app.state.settings
