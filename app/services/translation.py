from typing import Dict, List, Generator
from app.clients.qwen_client import QwenClient


class TranslationService:
    """翻译服务：中译英、英译中"""
    def __init__(self, client: QwenClient) -> None:
        self.client = client

    def zh_to_en(self, text: str) -> str:
        """将中文翻译为英文。"""
        system_prompt = (
            "You are a professional translation assistant. Translate the user's text accurately and naturally. "
            "Keep formatting, numbers, and special terms. Output only the translated text."
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + " Source language: Chinese. Target language: English."},
            {"role": "user", "content": text.strip()},
        ]
        return self.client.chat(messages)

    def en_to_zh(self, text: str) -> str:
        """将英文翻译为中文。"""
        system_prompt = (
            "You are a professional translation assistant. Translate the user's text accurately and naturally. "
            "Keep formatting, numbers, and special terms. Output only the translated text."
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + " Source language: English. Target language: Chinese."},
            {"role": "user", "content": text.strip()},
        ]
        return self.client.chat(messages)

    def zh_to_en_stream(self, text: str) -> Generator[str, None, None]:
        """流式：中文 -> 英文。返回纯文本片段，由路由封装为 SSE。"""
        system_prompt = (
            "You are a professional translation assistant. Translate the user's text accurately and naturally. "
            "Keep formatting, numbers, and special terms. Output only the translated text."
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + " Source language: Chinese. Target language: English."},
            {"role": "user", "content": text.strip()},
        ]
        for chunk in self.client.chat_stream(messages):
            yield chunk

    def en_to_zh_stream(self, text: str) -> Generator[str, None, None]:
        """流式：英文 -> 中文。返回纯文本片段，由路由封装为 SSE。"""
        system_prompt = (
            "You are a professional translation assistant. Translate the user's text accurately and naturally. "
            "Keep formatting, numbers, and special terms. Output only the translated text."
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + " Source language: English. Target language: Chinese."},
            {"role": "user", "content": text.strip()},
        ]
        for chunk in self.client.chat_stream(messages):
            yield chunk