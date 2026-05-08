import torch
from src.model import load_model_and_tokenizer


class InferenceEngine:
    def __init__(self, model_name="gpt2", adapter_path=None):
        self.model, self.tokenizer = load_model_and_tokenizer(model_name, adapter_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def answer(self, context, question, max_new_tokens=50):
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_answer = generated_text[len(input_text):].strip()
        return predicted_answer
