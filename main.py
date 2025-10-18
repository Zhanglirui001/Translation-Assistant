from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, field_validator
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Any, Dict, List, Optional
try:
    import dashscope
    from dashscope import Generation, Chat
except Exception:
    dashscope = None
    Generation = None
    Chat = None


def _clean_env_value(value: str | None, default: str = "") -> str:
    """清理 .env 值，去除内联注释与首尾空白。"""
    if value is None:
        return default
    return value.split("#", 1)[0].strip()


class QwenSettings(BaseModel):
    """统一管理大模型相关的密钥与参数。"""
    api_key: str
    model: str
    base_url: str
    timeout: int = 30

    @field_validator("api_key")
    @classmethod
    def api_key_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("QWEN_API_KEY 未配置，请在 .env 中设置")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timeout 必须为正整数")
        return v


def load_settings() -> QwenSettings:
    """从 .env 加载配置并返回 QwenSettings。"""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    timeout_str = _clean_env_value(os.getenv("Timeout"), "30")
    settings = QwenSettings(
        api_key=_clean_env_value(os.getenv("QWEN_API_KEY")),
        model=_clean_env_value(os.getenv("QWEN_MODEL"), "qwen-turbo"),
        base_url=_clean_env_value(os.getenv("QWEN_BASE_URL"), "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=int(timeout_str),
    )
    return settings


app = FastAPI(title="FastAPI Demo with Config")


class QwenClient:
    """通义千问客户端封装，统一对接 dashscope SDK。"""

    def __init__(self, settings: QwenSettings) -> None:
        self.settings = settings
        if dashscope is None:
            raise RuntimeError("dashscope SDK 未安装或导入失败")
        dashscope.api_key = settings.api_key

    def _extract_text(self, resp: Any) -> str:
        """尽量从响应中提取文本内容。"""
        # 常见：对象有 output_text 属性
        if hasattr(resp, "output_text"):
            try:
                return str(resp.output_text)
            except Exception:
                pass
        # 兼容字典或包含 output/choices/message 的结构
        try:
            output = getattr(resp, "output", None)
            if output is None and isinstance(resp, dict):
                output = resp.get("output")
            if isinstance(output, dict):
                choices = output.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        return str(message.get("content", ""))
                    # 某些模型可能直接返回 text
                    text = choices[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        # 兜底：转换为字符串
        return str(resp)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """通用单轮文本生成。"""
        try:
            resp = Generation.call(
                model=self.settings.model,
                prompt=prompt,
                timeout=self.settings.timeout,
                **kwargs,
            )
            return self._extract_text(resp)
        except Exception as e:
            logger.error("QwenClient.generate 调用失败: {}", e)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """多轮对话（OpenAI 风格 messages）。"""
        try:
            resp = Chat.call(
                model=self.settings.model,
                messages=messages,
                timeout=self.settings.timeout,
                **kwargs,
            )
            return self._extract_text(resp)
        except Exception as e:
            logger.error("QwenClient.chat 调用失败: {}", e)
            raise

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """使用对话接口实现可靠翻译。"""
        system_prompt = (
            "You are a professional translation assistant. Translate the user's text accurately and naturally. "
            "Keep formatting, numbers, and special terms. Output only the translated text."
        )
        if source_lang:
            system_prompt += f" Source language: {source_lang}."
        system_prompt += f" Target language: {target_lang}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        return self.chat(messages)


def get_qwen_client() -> QwenClient:
    client = getattr(app.state, "qwen_client", None)
    if client is None:
        raise RuntimeError("QwenClient 未初始化，请检查应用启动流程是否成功加载配置。")
    return client


@app.on_event("startup")
def on_startup() -> None:
    settings = load_settings()
    app.state.settings = settings  # 将配置挂载至全局应用状态，供各路由/服务使用

    # 初始化通义千问客户端
    if dashscope is not None:
        app.state.qwen_client = QwenClient(settings)
    else:
        logger.warning("dashscope SDK 未可用，QwenClient 未初始化。")

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


# 提供一个便捷的获取配置的函数，便于在依赖或业务模块中调用
# 使用方式：from main import get_settings
# 然后在路由/服务中调用 settings = get_settings()

def get_settings() -> QwenSettings:
    return app.state.settings