from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


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
    verify_ssl: bool = True

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
    """从项目根目录的 .env 加载配置并返回 QwenSettings。"""
    # 相对于 app/ 目录的上一级，即项目根目录
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    timeout_str = _clean_env_value(os.getenv("Timeout"), "30")
    verify_ssl_str = _clean_env_value(os.getenv("QWEN_VERIFY_SSL"), "true")
    settings = QwenSettings(
        api_key=_clean_env_value(os.getenv("QWEN_API_KEY")),
        model=_clean_env_value(os.getenv("QWEN_MODEL"), "qwen-turbo"),
        base_url=_clean_env_value(os.getenv("QWEN_BASE_URL"), "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=int(timeout_str),
        verify_ssl=verify_ssl_str.lower() in ("1", "true", "yes", "y", "on"),
    )
    return settings