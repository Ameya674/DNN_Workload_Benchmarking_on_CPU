import onnx
from onnxsim import simplify
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
# from sample_mlp import SimpleMLP

####################################################################
# For Part 1 and Part 2, you can replace the commented out code below
# with your model code and then use this file for exporting onnx. 
# For Part 3 and Part 4, you can copy torch.onnx.export command inside your main 
# file, where you downloaded and run the model. Make sure the variable
# named 'model' contains the model object. For Part 4 use opset version 
# 17 (opset_version=17) to export the model

#####################################################################
# class SimpleMLP(nn.Module):
#     def __init__(self):
#         super(SimpleMLP, self).__init__()
#         self.layer1 = nn.Linear(784, 512)
#         self.hidden_layer1 = nn.Linear(512, 512)
#         self.output_layer = nn.Linear(512, 10)

#     def forward(self, x):
#       x = x.view(-1, 784)  
#       layer1_output = self.layer1(x)
#       hidden_layer1_output = F.relu(self.hidden_layer1(layer1_output))
#       output = F.softmax(self.output_layer(hidden_layer1_output), dim = 1)
#       return output

# model = SimpleMLP()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5) # input channels = 1, output channels = 1, kernel_size = 3
        self.pool1 = nn.MaxPool2d(2, 2) # kernel_size = 2, stride = 2
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 4 * 4, 16)
        self.fc2 = nn.Linear(16, 10)


    def forward(self, x):
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.softmax(self.fc2(x), dim = -1)
      return x


model = SimpleCNN()

####################################################################
# Create a random input for the onnnx export script

batch_size = 1

data = torch.rand(batch_size,1,28,28)

torch.onnx.export(model,                     # model being run
                data,            # model input (or a tuple for multiple inputs)
                "onnx_exported_model.onnx",           # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['ip'],  # the model's input names
                output_names = ['op'],      # the model's output names
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

####################################################################
# Simplify the model using onnxsim

input_model_path = "onnx_exported_model.onnx"  # Replace with your model's path
output_model_path = "CNN.onnx"

# Load the ONNX model
model = onnx.load(input_model_path)

# Simplify the model (includes constant propagation)
simplified_model, check = simplify(model)

# Check if simplification was successful
if check:
    print("Model simplification successful. Saving the simplified model...")
    # Save the simplified model
    onnx.save(simplified_model, output_model_path)
    print(f"Simplified model saved at: {output_model_path}")
else:
    print("Model simplification failed.")
