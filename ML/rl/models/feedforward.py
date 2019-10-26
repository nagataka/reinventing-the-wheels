import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math

class Feedforward(nn.Module):
    def __init__(self, in_size, out_size):
        super(Feedforward, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        return F.softmax(self.fc(x))


class ValueNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.fc(x)


class GaussianConvPolicy(nn.Module):
    def __init__(self, img_channels, out_size):
        super(GaussianConvPolicy, self).__init__()
        # parameters to determine mu for each action value (steer, gas, )

        # CarRacing-v0: input = (3x96x96)
        self.conv1 = nn.Conv2d(img_channels, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2)
        
        flattened_size = 4*4*128

        self.li11 = nn.Linear(flattened_size, 1)
        self.li12 = nn.Linear(flattened_size, 1)
        self.li13 = nn.Linear(flattened_size, 1)

        self.li21 = nn.Linear(flattened_size, 1)
        self.li22 = nn.Linear(flattened_size, 1)
        self.li23 = nn.Linear(flattened_size, 1)

    def get_action(self, mu, sigma):
        dist = MultivariateNormal(mu, sigma)
        val = dist.sample()
        log_prob = dist.log_prob(val)
        return val, log_prob

    def forward(self, x):
        device = x.device
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)

        mu1 = self.li11(x)
        mu2 = self.li12(x)
        mu3 = self.li13(x)
        Mu =  torch.cat((mu1, mu2, mu3), 1)
        
        sig1 = torch.exp(self.li21(x))
        sig2 = torch.exp(self.li22(x))
        sig3 = torch.exp(self.li23(x))

        I = torch.eye(3).to(device)
        sig = torch.cat((sig1, sig2, sig3), 1)
        Sigma = I*sig
        action, log_prob = self.get_action(Mu, Sigma)

        return action, log_prob
