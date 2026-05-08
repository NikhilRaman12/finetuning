import json
import torch
import evaluate
from datasets import load_dataset
from src.inference import InferenceEngine


def run_squad_eval(adapter_path="./healthcare-gpt", num_examples=1000):
    squad_metric = evaluate.load("squad")
    dataset = load_dataset("squad")
    original_eval_examples = dataset["validation"].select(range(num_examples))

    engine = InferenceEngine(adapter_path=adapter_path)

    predictions = []
    references = []

    for example in original_eval_examples:
        question = example["question"]
        context = example["context"]
        original_answers = example["answers"]

        predicted_answer = engine.answer(context, question, max_new_tokens=50)

        predictions.append({"prediction_text": predicted_answer, "id": example["id"]})
        references.append({"answers": original_answers, "id": example["id"]})

    results = squad_metric.compute(predictions=predictions, references=references)

    print("\nSQuAD Evaluation Results:")
    print(f"Exact Match: {results['exact_match']}")
    print(f"F1 Score: {results['f1']}")

    output_json_file = "squad_evaluation_results.json"
    with open(output_json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {output_json_file}")

    return results


if __name__ == "__main__":
    run_squad_eval()
