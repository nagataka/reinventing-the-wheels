import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.original_gan import Generator, Discriminator
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

z_dim = 100
img_shape = (28, 28) # MNIST
epochs = 200
batch = 100
lr = 0.0002

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
            # Train the discriminator
            discriminator.zero_grad()
            z = torch.randn(batch, z_dim).to(device)
            generated_imgs = generator(z)
            pred_real = discriminator( data.view(-1, img_shape[0]*img_shape[1]).to(device) )
            pred_fake = discriminator( generated_imgs.detach() )
            loss_real = loss(pred_real, torch.ones(batch, 1).to(device) )
            loss_fake = loss(pred_fake, torch.zeros(batch, 1).to(device) )
            loss_D = (loss_real+loss_fake)/2
            loss_D.backward()
            optim_d.step()

            # Train the generator
            generator.zero_grad()
            z = torch.randn(batch, z_dim).to(device)
            generated_imgs = generator(z)
            validity = torch.ones(batch, 1).to(device) # Generator wants the discriminator to say 'real'
            pred = discriminator( generated_imgs )
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
        save_image(data.view(data.size(0), 1, 28, 28), "images/test%d.png" % cur_epoch, nrow=10)
        save_image(generated_imgs.view(generated_imgs.size(0), 1, 28, 28), "images/epoch%d_N.png" % cur_epoch, nrow=10)

    writer.close()
    return generated_imgs

def main():
    # MNIST: (1, 28, 28) with the data range [from 0. to 1.]
    mnist = datasets.MNIST(
                './data/', 
                download=True, 
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
                    #[transforms.ToTensor()]
                ))
    data_loader = torch.utils.data.DataLoader(mnist,
                                              batch_size=batch,
                                              shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(batch, z_dim, img_shape).to(device)
    discriminator = Discriminator(batch, img_shape).to(device)

    gen_imgs = train(generator, discriminator, data_loader, device)


if __name__ == '__main__':
    main()
