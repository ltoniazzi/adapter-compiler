import torch
import torch.nn as nn
import onnxruntime as ort
import pathlib

ROOT_FOLDER = pathlib.Path(__file__).parent.parent

# Define the main model with conditional LoRA adapters
class ConditionalLoRAModel(nn.Module):
    def __init__(self):
        super(ConditionalLoRAModel, self).__init__()
        self.fc0 = nn.Linear(10, 20)

        # Linear with corresponding adapter 1
        self.fc1 = nn.Linear(20, 5)
        self.lora_adapter_1 = LoRAAdapter(20, 5)

        # Linear with corresponding adapter 2
        self.fc2 = nn.Linear(20, 2)
        self.lora_adapter_2 = LoRAAdapter(20, 2)


    def forward(self, x, adapter_number):
        x = self.fc0(x)
        
        # Swap adapter ad runtime
        if adapter_number == 1:
            x = self.fc1(x) + self.lora_adapter_1(x)
        elif adapter_number == 2:
            x = self.fc2(x) + self.lora_adapter_2(x)
        else:
            raise ValueError("adapter_number not valid")
        
        return x


# Define LoRA adapter
class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRAAdapter, self).__init__()
        self.lora = nn.Sequential(
            nn.Linear(in_features, rank, bias=False),
            nn.Linear(rank, out_features, bias=False)
        )

    def forward(self, x):
        return self.lora(x)
    

def export_model_with_adapters(model, dummy_input):
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


if __name__ == "__main__":

    models_folder = ROOT_FOLDER / "models"
    model_path = str(models_folder / "conditional_lora_model.onnx")

    export_model_with_adapters(model=ConditionalLoRAModel(), dummy_input=torch.randn(1, 10))

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
    
