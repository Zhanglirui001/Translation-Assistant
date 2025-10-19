from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from loguru import logger
from fastapi.responses import StreamingResponse
from fastapi import Query

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


class SubmitTranslateRequest(BaseModel):
    text: str
    direction: str  # zh_to_en | en_to_zh

    @field_validator("text")
    @classmethod
    def text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text 不能为空")
        return v.strip()

    @field_validator("direction")
    @classmethod
    def direction_valid(cls, v: str) -> str:
        v2 = (v or "").strip().lower()
        if v2 not in {"zh_to_en", "en_to_zh"}:
            raise ValueError("direction 仅支持 zh_to_en 或 en_to_zh")
        return v2

class SubmitSummarizeRequest(BaseModel):
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

# 异步任务提交：翻译
@router.post("/api/tasks/translate")
async def submit_translate(req: Request):
    data = await req.json()
    payload = SubmitTranslateRequest(**data)
    svc = req.app.state.translation_service
    tm = req.app.state.task_manager
    if payload.direction == "zh_to_en":
        work = svc.zh_to_en
    else:
        work = svc.en_to_zh
    task_id = tm.submit(work_fn=work, task_type=f"translate:{payload.direction}", params={"text": payload.text})
    return {"task_id": task_id}

# 异步任务提交：总结
@router.post("/api/tasks/summarize")
async def submit_summarize(req: Request):
    data = await req.json()
    payload = SubmitSummarizeRequest(**data)
    svc = req.app.state.summarization_service
    tm = req.app.state.task_manager
    task_id = tm.submit(
        work_fn=svc.summarize,
        task_type="summarize",
        params={"text": payload.text, "target_lang": payload.target_lang, "max_points": payload.max_points},
    )
    return {"task_id": task_id}

# 轮询任务状态与结果
@router.get("/api/tasks/status")
async def get_task_status(req: Request, task_id: str = Query(..., description="任务ID")):
    tm = req.app.state.task_manager
    data = tm.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="task 不存在")
    # 返回精简后的视图（避免泄漏敏感参数）
    return {
        "id": data["id"],
        "status": data["status"],
        "type": data["type"],
        "result": data.get("result"),
        "error": data.get("error"),
    }

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