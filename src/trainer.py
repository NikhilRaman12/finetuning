import math
from transformers import Trainer, TrainingArguments


def run_training(model, tokenizer, train_dataset, eval_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity: {perplexity}")

    return trainer, eval_results, perplexity
