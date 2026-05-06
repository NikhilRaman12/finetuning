# GPT-2 Fine-tuning with LoRA - Modular Production-Ready Code

A complete, modular, production-grade codebase for fine-tuning GPT-2 with LoRA on multiple tasks including **Question-Answering (SQuAD)** and **Instruction-Following**.

## 📋 Features

✨ **Modular Architecture**
- Separate modules for model, data, training, and inference
- Easy to extend and customize
- Production-ready code with logging and error handling

🎯 **Multiple Fine-tuning Modes**
- **QA Mode**: Question-Answering fine-tuning on SQuAD dataset
- **Instruction Mode**: Instruction-following fine-tuning (Alpaca-style or custom datasets)

⚙️ **LoRA Integration**
- Efficient parameter-efficient fine-tuning (PEFT)
- Reduced memory footprint
- Faster training

🚀 **Colab Compatible**
- GPU-optimized for T4 (free tier)
- Automatic dependency installation
- End-to-end notebook script included

📊 **Hugging Face Trainer**
- Built-in evaluation and logging
- Mixed precision training support
- Checkpoint saving and loading

## 🏗️ Project Structure

```
finetuning/
├── config.py                          # Centralized configuration
├── train.py                           # Main training script
├── inference.py                       # Inference script
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── src/
│   ├── __init__.py
│   ├── model.py                      # Model initialization with LoRA
│   ├── data.py                       # Dataset loading & preprocessing
│   ├── trainer.py                    # Training with HF Trainer
│   └── inference.py                  # Inference engine
└── examples/
    ├── colab_notebook.py             # Colab-ready end-to-end script
    └── custom_dataset_example.py     # Custom dataset examples
```

## 🚀 Quick Start

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/NikhilRaman12/finetuning.git
cd finetuning

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

#### QA Fine-tuning (SQuAD)
```bash
python train.py --mode qa
```

#### Instruction Fine-tuning
```bash
# Default (Alpaca dataset)
python train.py --mode instruction

# Custom dataset
python train.py --mode instruction --dataset_path path/to/dataset.csv
```

### 3. Inference

#### Interactive Mode
```bash
python inference.py --mode interactive
```

#### QA Inference
```bash
python inference.py --mode qa --input "What is machine learning?"
```

#### Instruction Inference
```bash
python inference.py --mode instruction --input "Summarize this text" --context "Text to summarize"
```

### 4. Google Colab

```bash
# Copy colab_notebook.py to Colab and run
!python examples/colab_notebook.py
```

## 📝 Configuration

All hyperparameters are centralized in `config.py`:

```python
# Model Configuration
MODEL_CONFIG = ModelConfig(
    model_name="gpt2",
    use_cache=False,
    torch_dtype="float32"
)

# LoRA Configuration
LORA_CONFIG = LoRAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none"
)

# Training Configuration
TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    # ... more configs
)
```

Modify these values before training to customize your setup.

## 📊 Dataset Format

### QA Format (SQuAD)
```
Q: What is machine learning?
A: Machine learning is a subset of AI...
```

### Instruction Format (Alpaca-style)
```csv
instruction,input,output
"Summarize this text","Lorem ipsum...","A summary"
"What is AI?","","Artificial Intelligence is..."
```

### Custom CSV Dataset
```csv
instruction,input,output
"Medical question","Patient symptoms","Medical advice"
"Diagnosis","Lab results","Possible conditions"
```

## 🧠 Model Architecture

- **Base Model**: GPT-2 (124M parameters)
- **LoRA**: 
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
  - Target modules: c_attn (query/key/value projections)

## ⚡ Training Tips

1. **For T4 GPU in Colab**:
   - Batch size: 8-16
   - Gradient accumulation: 1-2
   - Mixed precision: Disable for stability

2. **For larger GPUs**:
   - Increase batch size to 32+
   - Enable fp16 for faster training
   - Increase gradient accumulation if memory is tight

3. **Dataset preparation**:
   - Ensure clean, well-formatted data
   - Use max_length=512 for GPT-2
   - Balance train/eval split

## 📈 Monitoring Training

Training logs are saved in `./logs/` directory. View with TensorBoard:

```bash
tensorboard --logdir=./logs
```

## 💾 Model Checkpoints

Fine-tuned models are saved in `./results/` by default:

```
results/
├── checkpoint-500/
├── checkpoint-1000/
└── pytorch_model.bin
```

Load a specific checkpoint:
```python
from src.inference import InferenceEngine

inference = InferenceEngine(
    model_name="gpt2",
    adapter_path="./results/checkpoint-500"
)
```

## 🔧 Customization

### Use Different Base Model
```python
# In config.py
MODEL_CONFIG = ModelConfig(model_name="gpt2-medium")  # or "gpt2-large"
```

### Adjust LoRA Parameters
```python
# In config.py
LORA_CONFIG = LoRAConfig(
    r=16,  # Increase rank
    lora_alpha=32,
    lora_dropout=0.05
)
```

### Custom Training Parameters
```python
# In config.py
TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=5,
    learning_rate=1e-4,
    per_device_train_batch_size=16
)
```

## 🔍 Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Set `fp16=True` (for compatible GPUs)

### Slow Training
- Ensure GPU is being used: Check logs for "cuda"
- Increase batch size if memory permits
- Use `fp16=True` for faster computation

### Poor Results
- Train for more epochs
- Increase learning rate slightly
- Ensure dataset quality
- Use domain-specific data for better performance

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [SQuAD Dataset](https://huggingface.co/datasets/squad)

## 📄 License

MIT License

## 👨‍💻 Author

Nikhil Raman

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Fine-tuning! 🚀**
