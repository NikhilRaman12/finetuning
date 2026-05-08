from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-2 LoRA QA API", version="1.0.0")

# Lazy-load model on first request
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from src.inference import InferenceEngine
        adapter_path = os.getenv("ADAPTER_PATH", "./healthcare-gpt")
        _engine = InferenceEngine(adapter_path=adapter_path)
    return _engine


class QARequest(BaseModel):
    context: str
    question: str
    max_new_tokens: int = 50


class QAResponse(BaseModel):
    answer: str
    context: str
    question: str


@app.get("/")
def root():
    return {"status": "ok", "model": "GPT-2 LoRA fine-tuned on SQuAD"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/answer", response_model=QAResponse)
def answer(req: QARequest):
    if not req.context.strip() or not req.question.strip():
        raise HTTPException(status_code=400, detail="context and question must not be empty")
    try:
        engine = get_engine()
        ans = engine.answer(req.context, req.question, req.max_new_tokens)
        return QAResponse(answer=ans, context=req.context, question=req.question)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
