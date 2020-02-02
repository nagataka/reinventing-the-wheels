import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, batch, z_dim, out_shape, num_classes):
        super(Generator, self).__init__()

        self.batch_size = batch
        self.z_dim = z_dim
        self.out_shape = out_shape
        self.num_classes = num_classes
        
        """
        self.fc1 = nn.Linear(z_dim, z_dim*2)
        self.fc2 = nn.Linear(z_dim*2, z_dim*4)
        self.fc3 = nn.Linear(z_dim*4, z_dim*8)
        self.fc4 = nn.Linear(z_dim*8, out_shape[0]*out_shape[1])
        """
        self.fc1 = nn.Linear(z_dim+num_classes, 256) # simple concat
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2) # Try LeakyReLU instead?
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        print("Build Generator: ", self)

    def forward(self, z):
        x = self.relu( self.fc1(z) )
        x = self.relu( self.fc2(x) )
        x = self.relu( self.fc3(x) )
        #out = self.sigmoid(self.fc4(x))
        out = self.tanh(self.fc4(x))
        return out


class Discriminator(nn.Module):
    def __init__(self, batch, out_shape, num_classes):
        super(Discriminator, self).__init__()

        self.batch_size = batch
        self.out_shape = out_shape
        self.out_shape_total = out_shape[0]*out_shape[1]
        self.num_classes = num_classes

        """
        self.fc1 = nn.Linear(self.out_shape_total, self.out_shape_total//2)
        self.fc1 = nn.Linear(self.out_shape_total, self.out_shape_total//2)
        self.fc2 = nn.Linear(self.out_shape_total//2, self.out_shape_total//4)
        self.fc3 = nn.Linear(self.out_shape_total//4, self.out_shape_total//8)
        self.fc4 = nn.Linear(self.out_shape_total//8, 1)
        """
        self.fc1 = nn.Linear(self.out_shape_total + num_classes, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.out = nn.Sigmoid()

        print("Build Discriminator: ", self)

    def forward(self, x):
        x = self.leakyrelu( self.fc1(x) )
        x = F.dropout(x, 0.3)
        x = self.leakyrelu( self.fc2(x) )
        x = F.dropout(x, 0.3)
        x = self.leakyrelu( self.fc3(x) )
        x = F.dropout(x, 0.3)

        out = self.out( self.fc4(x) )

        return out
