from typing import Any, Dict, List, Optional, Generator
import json
import ssl
import time
import urllib.request as urlrequest

try:
    import requests
except Exception:
    requests = None

try:
    import dashscope
    from dashscope import Generation, Chat
except Exception:
    dashscope = None
    Generation = None
    Chat = None

from loguru import logger
from app.config import QwenSettings


class QwenClient:
    """通义千问客户端封装，统一对接 dashscope SDK。"""

    def __init__(self, settings: QwenSettings) -> None:
        self.settings = settings
        self.use_http_fallback = (dashscope is None or Generation is None or Chat is None)
        if not self.use_http_fallback:
            dashscope.api_key = settings.api_key
        else:
            logger.warning("dashscope SDK 不可用，启用HTTP兼容模式访问: {}", settings.base_url)

    def _extract_text(self, resp: Any) -> str:
        """尽量从响应中提取文本内容。"""
        # 常见：对象有 output_text 属性
        try:
            output_text = getattr(resp, "output_text", None)
        except Exception:
            output_text = None
        if output_text is not None:
            try:
                return str(output_text)
            except Exception:
                pass
        # 顶层 dict 直接含有 output_text
        if isinstance(resp, dict) and "output_text" in resp:
            try:
                return str(resp["output_text"])
            except Exception:
                pass
        # 兼容字典或包含 output/choices/message 的结构
        try:
            try:
                output = getattr(resp, "output", None)
            except Exception:
                output = None
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
            # 顶层 choices（某些兼容模式返回）
            if isinstance(resp, dict):
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    # OpenAI 兼容结构：choices[0].message.content
                    msg = choices[0].get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        return str(msg["content"]) 
                    # 或者直接 text 字段
                    if "text" in choices[0]:
                        return str(choices[0]["text"]) 
        except Exception:
            pass
        # 兜底：转换为字符串
        return str(resp)

    def _http_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """在 dashscope SDK 不可用时，使用兼容模式 HTTP 直接调用。"""
        url = self.settings.base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TranslationAssistant/1.0",
            "Authorization": f"Bearer {self.settings.api_key}",
            "Connection": "close",
            "Accept-Encoding": "identity",
        }
        data = json.dumps(payload, ensure_ascii=False)
        # 优先使用 requests（若可用），更稳健的 TLS 实现
        if requests is not None:
            last_err: Exception | None = None
            verify = self.settings.verify_ssl
            for attempt in range(3):
                try:
                    r = requests.post(
                        url,
                        headers=headers,
                        data=data.encode("utf-8"),
                        timeout=self.settings.timeout,
                        verify=verify,
                    )
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    last_err = e
                    # 如果是 SSL 错误，下一次尝试关闭校验
                    if attempt == 0 and verify:
                        verify = False
                    if attempt < 2:
                        time.sleep(0.5 * (2 ** attempt))
                    else:
                        raise last_err
        # 退回 urllib 实现
        req = urlrequest.Request(
            url,
            data=data.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        context = ssl.create_default_context() if self.settings.verify_ssl else ssl._create_unverified_context()
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urlrequest.urlopen(req, timeout=self.settings.timeout, context=context) as resp:
                    body = resp.read()
                    return json.loads(body.decode("utf-8"))
            except Exception as e:
                last_err = e
                # 若是 SSL 异常，自动切换为不校验的上下文重试一次
                if attempt == 0 and isinstance(e, ssl.SSLError):
                    context = ssl._create_unverified_context()
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    raise last_err

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """通用单轮文本生成。"""
        if self.use_http_fallback:
            payload: Dict[str, Any] = {"model": self.settings.model, "prompt": prompt}
            if kwargs:
                payload.update(kwargs)
            resp = self._http_request("completions", payload)
            return self._extract_text(resp)
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
        if self.use_http_fallback:
            payload: Dict[str, Any] = {"model": self.settings.model, "messages": messages}
            if kwargs:
                payload.update(kwargs)
            resp = self._http_request("chat/completions", payload)
            return self._extract_text(resp)
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

    # 新增：流式对话，返回文本片段生成器
    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """返回一个逐段文本生成器，若 SDK 不支持则模拟分片。"""
        # 简化实现：直接调用 chat 并把文本切片模拟
        text = self.chat(messages)
        # 按固定大小分片（保持与之前实现一致）
        size = 16
        for i in range(0, len(text), size):
            yield text[i:i+size]