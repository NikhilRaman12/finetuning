"""
HF Spaces entry point.
Runs FastAPI on port 8000 in a background thread,
then launches Gradio on port 7860 (the port HF Spaces exposes).
"""
import threading
import uvicorn
import gradio as gr
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app_api = FastAPI()
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from src.inference import InferenceEngine
        adapter_path = os.getenv("ADAPTER_PATH", "./healthcare-gpt")
        logger.info(f"Loading model from {adapter_path}")
        _engine = InferenceEngine(adapter_path=adapter_path)
    return _engine


class QARequest(BaseModel):
    context: str
    question: str
    max_new_tokens: int = 50


@app_api.get("/health")
def health():
    return {"status": "healthy"}


@app_api.post("/answer")
def answer(req: QARequest):
    if not req.context.strip() or not req.question.strip():
        raise HTTPException(status_code=400, detail="context and question must not be empty")
    engine = get_engine()
    ans = engine.answer(req.context, req.question, req.max_new_tokens)
    return {"answer": ans, "context": req.context, "question": req.question}


def run_api():
    uvicorn.run(app_api, host="0.0.0.0", port=8000, log_level="warning")


# ── Gradio ───────────────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
        "What is on top of the Main Building?",
    ],
    [
        "The Amazon rainforest covers most of the Amazon basin of South America.",
        "Where is the Amazon rainforest located?",
    ],
]


def ask(context: str, question: str, max_new_tokens: int) -> str:
    if not context.strip() or not question.strip():
        return "Please provide both a context and a question."
    try:
        resp = requests.post(
            "http://localhost:8000/answer",
            json={"context": context, "question": question, "max_new_tokens": int(max_new_tokens)},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["answer"]
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="GPT-2 LoRA QA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GPT-2 LoRA — Question Answering\nFine-tuned on SQuAD dataset")

    with gr.Row():
        with gr.Column(scale=2):
            context_box = gr.Textbox(label="Context", placeholder="Paste a paragraph here...", lines=6)
            question_box = gr.Textbox(label="Question", placeholder="Ask a question...", lines=2)
            max_tokens = gr.Slider(10, 150, value=50, step=10, label="Max new tokens")
            submit_btn = gr.Button("Get Answer", variant="primary")
        with gr.Column(scale=1):
            answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=[context_box, question_box])

    submit_btn.click(fn=ask, inputs=[context_box, question_box, max_tokens], outputs=answer_box)
    question_box.submit(fn=ask, inputs=[context_box, question_box, max_tokens], outputs=answer_box)


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("FastAPI started on port 8000")
    demo.launch(server_name="0.0.0.0", server_port=7860)
