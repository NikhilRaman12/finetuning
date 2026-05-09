import sys
import types

# pydub tries to import audioop/pyaudioop which doesn't exist in Python 3.13
# patch it with a dummy module before gradio imports pydub
dummy = types.ModuleType("audioop")
sys.modules["audioop"] = dummy
sys.modules["pyaudioop"] = dummy

import gradio as gr
from src.inference import InferenceEngine

engine = InferenceEngine(model_name="gpt2", adapter_path="Nikhil2003/healthcare-gpt")

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

def gradio_answer(context, question, max_new_tokens):
    if not context.strip() or not question.strip():
        return "Please provide both a context and a question."
    return engine.answer(context, question, max_new_tokens=int(max_new_tokens))

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
    submit_btn.click(fn=gradio_answer, inputs=[context_box, question_box, max_tokens], outputs=answer_box)
    question_box.submit(fn=gradio_answer, inputs=[context_box, question_box, max_tokens], outputs=answer_box)

demo.launch()
