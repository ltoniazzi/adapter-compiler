import torch
import onnxruntime as ort
import pathlib
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, PeftType


ROOT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, PeftType

# Define the LoRA Adapter class
class LoRAAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, r=4):
        super(LoRAAdapter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.r = r
        self.W_a = nn.Linear(input_dim, r, bias=False)
        self.W_b = nn.Linear(r, output_dim, bias=False)

    def forward(self, x):
        return self.W_b(self.W_a(x))

# Define the main model with conditional LoRA adapters
class ConditionalLoRAModel(nn.Module):
    def __init__(self, gpt_model, adapter_1_path, adapter_2_path):
        super(ConditionalLoRAModel, self).__init__()
        self.gpt_model = gpt_model
        
        # Assuming the target modules are named as per GPT-2 architecture
        self.lora_adapter_1 = torch.load(adapter_1_path)
        self.lora_adapter_2 = torch.load(adapter_2_path)

        

    def forward(self, x, adapter_number):
        outputs = self.gpt_model(inputs_embeds=x)

        # Assuming the adapters need to be applied to the last hidden state
        last_hidden_state = outputs.last_hidden_state

        if adapter_number == 1:
            for name, adapter in self.lora_adapter_1.items():
                last_hidden_state = adapter(last_hidden_state) + last_hidden_state
        elif adapter_number == 2:
            for name, adapter in self.lora_adapter_2.items():
                last_hidden_state = adapter(last_hidden_state) + last_hidden_state
        else:
            raise ValueError("adapter_number not valid")

        return last_hidden_state


def export_model_with_adapters(model, dummy_input, model_path):
    # Trun it into a graph
    scripted_model = torch.jit.script(model)

    # Dummy input for the adapter selection
    dummy_adapter_number = torch.tensor([1], dtype=torch.int)

    # Export model to onnx format
    torch.onnx.export(
        scripted_model, 
        (dummy_input, dummy_adapter_number), 
        model_path, 
        input_names=["input", "adapter_number"], 
        output_names=["output"], 
        dynamic_axes={
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}, 
            "adapter_number": {0: "batch_size"}
        }
    )

def add_adapters(model_name, adapters_paths):
    model = GPT2Model.from_pretrained(model_name)

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

    for adapter_path in adapters_paths:
        # Load the first adapter
        lora_model.load_state_dict(torch.load(adapter_path))

    ConditionalLoRAModel(model, adapters_paths[0], adapters_paths[1])

    # The model now has the second adapter loaded and ready for use
    print("Second LoRA adapter loaded.")





if __name__ == "__main__":

    models_folder = ROOT_FOLDER / "models"
    model_name = "gpt2"
    n_adapters = (1, 2)
    adapters_paths = [models_folder / f"{model_name}_lora_adapter_{num}.pt" for num in n_adapters]
    model_path = models_folder / f"{model_name}_with_adapters.onnx"

    model = add_adapters(model_name=model_name, adapters_paths=adapters_paths)



    export_model_with_adapters(model=model, dummy_input=torch.randn(1, 10))

    # Prepare inputs where int inputs that allow to swap the adapter
    dummy_input = torch.randn(1, 10)
    adapter_number_1 = torch.tensor([1], dtype=torch.int)
    adapter_number_2 = torch.tensor([2], dtype=torch.int)

    # Load onnx model with python runtime
    ort_session = ort.InferenceSession(model_path)

    # Collect inputs
    inputs_list = [
        {"input": dummy_input.numpy(), "adapter_number": adapter_number_1.numpy()},
        {"input": dummy_input.numpy(), "adapter_number": adapter_number_2.numpy()},
    ]

    for inputs in inputs_list:
        # Run inference
        outputs = ort_session.run(None, inputs)

        # Retrieve the output and check the output shape 
        print(f"\nModel output with adapter {inputs['adapter_number']}:", outputs[0])
    
