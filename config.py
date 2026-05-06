"""
Configuration module for fine-tuning GPT-2 with LoRA on healthcare/SQuAD dataset
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "gpt2"
    use_cache: bool = False  # Disable for gradient checkpointing
    torch_dtype: str = "float32"

@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "squad"
    max_length: int = 512
    test_size: float = 0.1
    seed: int = 42

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False  # Set to True if using GPU with fp16 support
    gradient_checkpointing: bool = True
    use_wandb: bool = False
    wandb_project: str = "gpt2-lora-finetuning"

# Main configuration
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
