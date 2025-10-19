from typing import Optional
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