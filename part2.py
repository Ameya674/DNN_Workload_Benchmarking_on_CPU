import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from fvcore.nn import FlopCountAnalysis
from thop import profile

n_epochs = 5  # Number of Epochs for training
batch_size = 1 # Batch size for training and testing TODO: Modify this variable to change batch size
log_interval = 100 # This variable manages how frequently do you want to print the training loss

####################################################################
# Avoid changing the below parameters 
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

####################################################################
# Train loader and test loader for the MNIST dataset
# This part of the code will download the MNIST dataset in the 
# same directory as this script. It will normalize the dataset and batch
# it into the batch size specified by batch_size var. 

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

####################################################################
# TODO: Define your model here
# See the example MLP linked in the lab document for help.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5) # input channels = 1, output channels = 1, kernel_size = 3
        self.pool1 = nn.MaxPool2d(2, 2) # kernel_size = 2, stride = 2
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(1, 1, 2)
        # self.pool3 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(1 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)


    def forward(self, x):
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      # x = self.pool3(self.conv3(x))
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.softmax(self.fc2(x), dim = -1)
      return x

network = SimpleCNN()
# Using the SGD (Stochastic Gradient Descent) optimizer
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []

####################################################################
# Train and test methods for training the model 

def train(epoch):
  network.train()
  total_training_time = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    batch_start_time = time.time()
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    batch_end_time = time.time()
    total_training_time += (batch_end_time - batch_start_time)
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  return total_training_time

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

####################################################################
# Train the model for given epochs

total_time = 0
for epoch in range(1, n_epochs + 1):
    time_per_epoch = train(epoch)
    total_time = total_time + time_per_epoch
    test()

print(f"Total Training time: {total_time}")

####################################################################
# Single inference

with torch.no_grad():
    test_iterator = iter(test_loader)
    data, target = next(test_iterator)
    single_batch_start = time.time()
    # Run single inference for 1000 times to avoid measurement overheads
    for i in range(0,1000):
      output = network(data)
    single_batch_end = time.time()

    single_batch_inf_time = (single_batch_end - single_batch_start)/1000
    print(f"Single Batch Inference time is {single_batch_inf_time} seconds for a batch size of {test_loader.batch_size}")


total_parameters = sum(p.numel() for p in network.parameters())
input_tensor = torch.randn(batch_size, 1, 28, 28)
macs = FlopCountAnalysis(network, input_tensor).total()
print("----------------------------------------")
print(f"Total number of parameters: {total_parameters}")
print(f"Total number of MACs: {macs}")

