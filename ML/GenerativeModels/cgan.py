import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.conditional_gan import Generator, Discriminator
from util import convert_onehot
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

z_dim = 100
img_shape = (28, 28) # MNIST
epochs = 200
batch = 100
lr = 0.0002
num_classes = 10

def train(generator, discriminator, data_loader, device):

    cur_epoch = 0

    # Define params and loss to potimize
    optim_g = Adam(generator.parameters(), lr=lr)
    optim_d = Adam(discriminator.parameters(), lr=lr)
    loss = nn.BCELoss()
    writer = SummaryWriter()

    while cur_epoch < epochs:
        print("Epoch: ", cur_epoch)
        for i, (data, label) in enumerate(data_loader):
            # Label is a list of int [0-9]. This needs to be one-hot for conditioning factor
            label = convert_onehot(label.view(label.shape[0], 1), num_classes).to(device)
            data = data.to(device)

            # Train the discriminator
            discriminator.zero_grad()
            z = torch.randn(batch, z_dim).to(device)
            generated_imgs = generator( torch.cat((z, label), dim=1) )
            pred_real = discriminator( torch.cat((data.view(-1, img_shape[0]*img_shape[1]), label), dim=1) )
            pred_fake = discriminator( torch.cat((generated_imgs.detach(), label), dim=1) )
            loss_real = loss(pred_real, torch.ones(batch, 1).to(device) )
            loss_fake = loss(pred_fake, torch.zeros(batch, 1).to(device) )
            loss_D = (loss_real+loss_fake)/2
            loss_D.backward()
            optim_d.step()

            # Train the generator
            generator.zero_grad()
            z = torch.randn(batch, z_dim).to(device)
            generated_imgs = generator( torch.cat((z, label), dim=1) )
            validity = torch.ones(batch, 1).to(device) # Generator wants the discriminator to say 'real'
            pred = discriminator( torch.cat((generated_imgs, label), dim=1) )
            loss_G = loss(pred, validity)
            loss_G.backward()
            optim_g.step()
            

        print("Loss G", loss_G.item())
        print("Loss D", loss_D.item())
        writer.add_scalars('runs/losses',{
                            "G": loss_G.item(),
                            "D": loss_D.item()}, cur_epoch)
        cur_epoch += 1
        print("Save image of epoch ", cur_epoch)
        save_image(data.view(data.size(0), 1, 28, 28), "images/cgan_test%d.png" % cur_epoch, nrow=10)
        save_image(generated_imgs.view(generated_imgs.size(0), 1, 28, 28), "images/cgan_epoch%d.png" % cur_epoch, nrow=10)

    writer.close()
    return generated_imgs

def main():
    # MNIST: (1, 28, 28) with the data range [from 0. to 1.]
    mnist = datasets.MNIST(
                './data/', 
                download=True, 
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
                ))
    data_loader = torch.utils.data.DataLoader(mnist,
                                              batch_size=batch,
                                              shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(batch, z_dim, img_shape, num_classes).to(device)
    discriminator = Discriminator(batch, img_shape, num_classes).to(device)

    gen_imgs = train(generator, discriminator, data_loader, device)


if __name__ == '__main__':
    main()
