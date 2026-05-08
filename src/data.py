from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def load_and_tokenize(tokenizer, train_size=5000, eval_size=1000):
    dataset = load_dataset("squad")

    def tokenize_function(examples):
        inputs = []
        for i in range(len(examples["question"])):
            context = examples["context"][i]
            question = examples["question"][i]
            answer = examples["answers"][i]["text"][0] if examples["answers"][i]["text"] else ""
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}{tokenizer.eos_token}"
            inputs.append(input_text)

        tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=tokenizer.model_max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    train_dataset = tokenized_datasets["train"].select(range(train_size))
    eval_dataset = tokenized_datasets["validation"].select(range(eval_size))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return train_dataset, eval_dataset, data_collator, dataset
