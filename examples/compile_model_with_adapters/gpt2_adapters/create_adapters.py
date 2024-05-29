import torch
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, PeftType
import pathlib

ROOT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent

models_folder = ROOT_FOLDER / "models"

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define the configuration for LoRA
config = LoraConfig(
    peft_type=PeftType.LORA,
    task_type="CAUSAL_LM",
    r=4,  # Low-rank parameter
    lora_alpha=32,  # Scaling parameter
    lora_dropout=0.1,  # Dropout rate
    target_modules=["c_attn", "c_proj"]  # Apply LoRA to all linear layers
)

# Apply LoRA to the model
lora_model = get_peft_model(model, config)

# Save the LoRA adapter
adapter_1_path = models_folder / f"{model_name}_lora_adapter_1.pt"
torch.save(lora_model.state_dict(), adapter_1_path)

# Assuming some training or further modifications happen here

# Save another LoRA adapter after modifications
adapter_2_path = models_folder / f"{model_name}_lora_adapter_2.pt"
torch.save(lora_model.state_dict(), adapter_2_path)

print(f"LoRA adapters saved to {adapter_1_path} and {adapter_2_path}")
