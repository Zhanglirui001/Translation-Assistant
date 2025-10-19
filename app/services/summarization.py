from typing import Optional, Iterable
from app.clients.qwen_client import QwenClient


class SummarizationService:
    """总结服务：精简长文本"""
    def __init__(self, client: QwenClient) -> None:
        self.client = client

    def summarize(self, text: str, target_lang: Optional[str] = None, max_points: int = 5) -> str:
        """对长文本进行精简总结，可指定目标语言与要点数量。"""
        system_prompt = (
            "You are a professional summarization assistant. Summarize the user's text into a concise form. "
            "Focus on key points, facts, numbers, and dates. Remove redundancy and filler. "
            "If a target language is specified, output in that language; otherwise, keep the original language. "
            f"Limit to about {max_points} bullet points or a short paragraph. "
            "Output only the summary text."
        )
        if target_lang:
            system_prompt += f" Target language: {target_lang}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        return self.client.chat(messages)

    # 流式总结：返回纯文本片段，由路由层统一包装为 SSE
    def summarize_stream(self, text: str, target_lang: Optional[str] = None, max_points: int = 5) -> Iterable[str]:
        system_prompt = (
            "You are a professional summarization assistant. Summarize the user's text into a concise form. "
            "Focus on key points, facts, numbers, and dates. Remove redundancy and filler. "
            "If a target language is specified, output in that language; otherwise, keep the original language. "
            f"Limit to about {max_points} bullet points or a short paragraph. "
            "Output only the summary text."
        )
        if target_lang:
            system_prompt += f" Target language: {target_lang}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        for piece in self.client.chat_stream(messages):
            if piece:
                yield piece