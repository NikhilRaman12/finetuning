---
title: GPT-2 LoRA QA
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: spaces_app.py
pinned: false
---

# GPT-2 LoRA — Question Answering

Fine-tuned GPT-2 with LoRA on the SQuAD dataset.

## Usage

Paste any paragraph as **Context**, type a **Question**, and click **Get Answer**.

## Model Details

- Base: `gpt2` (124M parameters)
- Fine-tuning: LoRA (r=8, alpha=16, target: `c_attn`)
- Dataset: SQuAD v1.1
- Perplexity: ~14.9
