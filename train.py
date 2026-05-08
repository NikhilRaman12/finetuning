from src.model import load_model_and_tokenizer
from src.data import load_and_tokenize
from src.trainer import run_training


def main():
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset, data_collator, _ = load_and_tokenize(tokenizer)
    trainer, eval_results, perplexity = run_training(model, tokenizer, train_dataset, eval_dataset, data_collator)

    print("Attempting to save model and tokenizer...")
    trainer.model.save_pretrained("./healthcare-gpt")
    tokenizer.save_pretrained("./healthcare-gpt")
    print("Model and tokenizer save commands executed.")


if __name__ == "__main__":
    main()
