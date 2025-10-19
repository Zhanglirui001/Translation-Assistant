from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from loguru import logger
from fastapi.responses import StreamingResponse

router = APIRouter()


class TranslateRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        return v.strip()


class SummarizeRequest(BaseModel):
    text: str
    target_lang: Optional[str] = None
    max_points: int = 5

    @field_validator("text")
    @classmethod
    def text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        return v.strip()

    @field_validator("max_points")
    @classmethod
    def points_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_points 必须为正整数")
        return v


# 翻译接口：中文 -> 英文（SSE 流式）
@router.post("/api/translate/zh-to-en")
async def api_translate_zh_to_en(req: Request):
    svc = req.app.state.translation_service
    try:
        ct = (req.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct == "text/plain":
            raw = await req.body()
            text = raw.decode("utf-8").strip()
        else:
            data = await req.json()
            text = (data.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text 不能为空")
        def _iter():
            for piece in svc.zh_to_en_stream(text):
                yield f"data: {piece}\n\n"
            yield "event: end\n\ndata: [DONE]\n\n"
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(_iter(), media_type="text/event-stream", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("api_translate_zh_to_en 调用失败: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# 翻译接口：英文 -> 中文（SSE 流式）
@router.post("/api/translate/en-to-zh")
async def api_translate_en_to_zh(req: Request):
    svc = req.app.state.translation_service
    try:
        ct = (req.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct == "text/plain":
            raw = await req.body()
            text = raw.decode("utf-8").strip()
        else:
            data = await req.json()
            text = (data.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text 不能为空")
        def _iter():
            for piece in svc.en_to_zh_stream(text):
                yield f"data: {piece}\n\n"
            yield "event: end\n\ndata: [DONE]\n\n"
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(_iter(), media_type="text/event-stream", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("api_translate_en_to_zh 调用失败: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# 总结接口：精简长文本（SSE 流式）
@router.post("/api/summarize")
async def api_summarize(req: Request):
    svc = req.app.state.summarization_service
    try:
        ct = (req.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct == "text/plain":
            raw = await req.body()
            text = raw.decode("utf-8").strip()
            target_lang = None
            max_points = 5
        else:
            data = await req.json()
            text = (data.get("text") or "").strip()
            target_lang = data.get("target_lang")
            max_points = int(data.get("max_points") or 5)
        if not text:
            raise HTTPException(status_code=422, detail="text 不能为空")
        def _iter():
            for piece in svc.summarize_stream(text, target_lang=target_lang, max_points=max_points):
                yield f"data: {piece}\n\n"
            yield "event: end\n\ndata: [DONE]\n\n"
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(_iter(), media_type="text/event-stream", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("api_summarize 调用失败: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# 常规对话接口：支持 text/plain 与 JSON（SSE 流式）
@router.post("/api/chat")
async def api_chat(req: Request):
    svc = req.app.state.chat_service
    try:
        ct = (req.headers.get("content-type") or "").split(";")[0].strip().lower()
        messages: List[Dict[str, str]] = []
        if ct == "text/plain":
            raw = await req.body()
            text = raw.decode("utf-8").strip()
            if not text:
                raise HTTPException(status_code=422, detail="text 不能为空")
            messages = [{"role": "user", "content": text}]
        else:
            data = await req.json()
            if isinstance(data.get("messages"), list) and data.get("messages"):
                for m in data["messages"]:
                    role = (m.get("role") or "user").strip()
                    content = (m.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
            else:
                text = (data.get("text") or "").strip()
                if not text:
                    raise HTTPException(status_code=422, detail="text 不能为空")
                system_msg = (data.get("system") or "").strip()
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                messages.append({"role": "user", "content": text})

        if not messages:
            raise HTTPException(status_code=422, detail="messages 不能为空")

        def _iter():
            for piece in svc.chat_stream(messages):
                yield f"data: {piece}\n\n"
            yield "event: end\n\ndata: [DONE]\n\n"

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(_iter(), media_type="text/event-stream", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("api_chat 调用失败: {}", e)
        raise HTTPException(status_code=500, detail=str(e))