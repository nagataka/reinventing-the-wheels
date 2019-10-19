# This code based on PyTorch official tutorial
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#q-network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class CNN(nn.Module):
    def __init__(self, in_channel, H, W, num_out):
	    super(CNN, self).__init__()
	    num_filters = in_channel*5
	    self.conv1 = nn.Conv2d(in_channel, num_filters, kernel_size=5)
	    self.bn1   = nn.BatchNorm2d(num_filters)
	    self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=5)
	    self.bn2   = nn.BatchNorm2d(num_filters*2)
	    self.conv3 = nn.Conv2d(num_filters, num_filters*3, kernel_size=5)
	    self.bn3   = nn.BatchNorm2d(num_filters*3)

	    # Number of Linear input connections depends on output of conv2d layers
	    # and therefore the input image size, so compute it.
	    def conv2d_size_out(size, kernel_size = 5, stride = 2):
	        return (size - (kernel_size - 1) - 1) // stride  + 1
	    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(W)))
	    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(H)))
	    linear_input_size = convw * convh * 32
	    self.head = nn.Linear(linear_input_size, num_out)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
