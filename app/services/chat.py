from typing import Generator, Dict, List
from loguru import logger
from app.clients.qwen_client import QwenClient


class ChatService:
    """常规对话服务，负责将客户端的流式输出封装为 SSE。"""

    def __init__(self, client: QwenClient) -> None:
        self.client = client

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        """将用户文本发送到模型，按片段返回文本（SSE 数据行格式）。"""
        if not text or not text.strip():
            # 直接返回一个空片段，前端可自行忽略
            yield "data: \n\n"
            return
        messages: List[Dict[str, str]] = [{"role": "user", "content": text.strip()}]
        try:
            for chunk in self.client.chat_stream(messages):
                # 将纯文本片段包装为 SSE 数据行
                yield f"data: {chunk}\n\n"
        except Exception as e:
            # 出错时通过 SSE 通知前端
            logger.error("ChatService.chat_stream 失败: {}", e)
            yield f"data: [ERROR] {e}\n\n"
        # 结束标记，便于前端停止读取
        yield "data: [DONE]\n\n"