import torch
import torch.nn as nn

# Define a more complex model with nested modules
class NestedModel(nn.Module):
    def __init__(self):
        super(NestedModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 15)
        )
        self.layer3 = nn.Linear(15, 10)
        self.layer4 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Instantiate the model
model = NestedModel()

# Example input tensor
input_tensor = torch.randn(1, 10)

# Loop through each top-level layer and apply the input
output = input_tensor
print("Using children():")
for layer in model.children():
    output = layer(output)
    print(output)

# Reset output for the next loop
output = input_tensor

# Loop through all modules, including nested ones
print("\nUsing modules():")
for layer in model.modules():
    if layer != model:  # Skip the model itself
        output = layer(output)
        print(output)
