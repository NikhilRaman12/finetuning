---
title: GPT-2 LoRA QA
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# GPT-2 LoRA — Question Answering

Fine-tuned GPT-2 with LoRA on the SQuAD dataset.

- Base model: `gpt2` (124M parameters)
- LoRA: r=8, alpha=16, target: `c_attn`
- Dataset: SQuAD v1.1
- Perplexity: ~14.89
