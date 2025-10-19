from typing import Iterable, Dict, List
from loguru import logger
from app.clients.qwen_client import QwenClient


class ChatService:
    """常规对话服务，返回纯文本片段；SSE 封装由路由层负责。"""

    def __init__(self, client: QwenClient) -> None:
        self.client = client

    def chat_stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        """接收 OpenAI 风格 messages，调用上游流式接口并逐片返回纯文本。"""
        # 规范化与过滤空内容
        norm_msgs: List[Dict[str, str]] = []
        for m in messages or []:
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            if content:
                norm_msgs.append({"role": role, "content": content})
        if not norm_msgs:
            return []  # 无内容则返回空迭代
        try:
            for chunk in self.client.chat_stream(norm_msgs):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error("ChatService.chat_stream 失败: {}", e)
            raise