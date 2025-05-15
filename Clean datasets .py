#semi-fictitious WGAN for clean datasets

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.autograd as autograd
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import os
import random
import torchvision
import torch.nn.init as init

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


DIM = 64 
Z_DIM=128
BATCH_SIZE = 256 
lambda_gp=10
num_epochs=100
total_weight=0.5
epoch_recording=30

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
if not os.path.exists('./save_dir'):
    os.mkdir('./save_dir')

transforms = transforms.Compose([
    transforms.Resize([64,64]),    
    transforms.ToTensor()          
]) 


path = './data_dir'

data_train = datasets.ImageFolder(path, transform=transforms)
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noises = torch.randn(BATCH_SIZE, Z_DIM).to(device)
 

class WGAN_D(nn.Module): 
    def __init__(self):
        super(WGAN_D, self).__init__()
        self.conv1=nn.Conv2d(3,DIM,4,2,1)
        self.BN1=nn.InstanceNorm2d(DIM,affine=True)
        self.conv2=nn.Conv2d(DIM,2*DIM,4,2,1)
        self.BN2=nn.InstanceNorm2d(2*DIM,affine=True)
        self.LR=nn.LeakyReLU(0.2)
        self.conv3=nn.Conv2d(2*DIM,4*DIM,4,2,1)
        self.BN3=nn.InstanceNorm2d(4*DIM,affine=True)
        self.linear1=nn.Linear(8*8*4*DIM,1)
    def forward(self, x):
        x=x.reshape(-1,3,64,64)
        x=self.conv1(x)
        x=self.LR(x)
        x=self.BN1(x)
        x=self.conv2(x)
        x=self.LR(x)
        x=self.BN2(x)
        x=self.conv3(x)
        x=self.LR(x)
        x=self.BN3(x)
        x = x.view(x.size(0),-1)
        x=self.linear1(x)
        return x.reshape(-1)
 
 
class WGAN_G(nn.Module): 
    def __init__(self):
        super(WGAN_G, self).__init__()
        self.linear=nn.Linear(128,8*8*4*DIM)
        self.relu=nn.ReLU(True)
        self.BN1=nn.BatchNorm1d(8*8*4*DIM)
        self.convt1=nn.ConvTranspose2d(4*DIM,2*DIM,4,2,padding=1)
        self.BN2= nn.BatchNorm2d(2*DIM)
        self.convt2=nn.ConvTranspose2d(2*DIM,DIM,4,2,padding=1)
        self.BN3= nn.BatchNorm2d(DIM)
        self.convt3=nn.ConvTranspose2d(DIM,3,4,2,padding=1)
        self.tanh=nn.Tanh()
    def forward(self, x):
        x = self.linear(x)
        x=self.relu(x)
        x=self.BN1(x)
        x = x.view(x.size(0),4*DIM,8,8)
        x=self.convt1(x)
        x=self.relu(x)
        x=self.BN2(x)
        x=self.convt2(x)
        x=self.relu(x)
        x=self.BN3(x)
        x=self.convt3(x)
        x=self.tanh(x)
        return x
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) in [nn.BatchNorm2d,nn.InstanceNorm2d]:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0) 



def compute_gradient_penalty(D,real_samples,fake_samples):
    eps = torch.rand(BATCH_SIZE, 1,1,1).to(device)
    eps= eps.expand(real_samples.size())
    X_inter = (eps * real_samples + ((1-eps)*fake_samples)).requires_grad_(True)
    d_interpolates = D(X_inter)
    fake=torch.ones(d_interpolates.size(),device=device)
    gradients = autograd.grad(outputs=d_interpolates, 
                              inputs=X_inter,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True
                              )[0]
    gradients = gradients.view(gradients.size(0),-1)
    gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2) * lambda_gp
    return gradient_penaltys


def train(D,G,outdir,z_dimension,num_epochs):
    d_optimizer = torch.optim.Adam(D.parameters(),lr=1e-4) 
    g_optimizer = torch.optim.Adam(G.parameters(),lr=1e-4)
    one = torch.ones(BATCH_SIZE).to(device)
    mone = -1*one .to(device)
    fake0=[]
    fake0=torch.tensor(fake0)

    for epoch in range(num_epochs):
        for i,(img,_) in enumerate(train_loader):
            num_img = img.size(0)
            real_img = img.to(device)
            for ii in range(5): 
                #Use current generated samples
                if random.random()>total_weight:
                    d_optimizer.zero_grad() 
                    real_out = D(real_img)
                    real_out.backward(mone)
                    z = torch.randn(num_img,z_dimension).to(device)
                    fake_img = G(z) 
                    fake_out = D(fake_img) 
                    fake_out.backward(one)
                    gradient_penalty = compute_gradient_penalty(D,real_img.data,fake_img.data)
                    gradient_penalty.backward(one)
                    WD=real_out.mean()-fake_out.mean()
                    d_loss = -real_out+fake_out+gradient_penalty
                    d_optimizer.step()
                #Use historical generated samples
                if epoch>0 and random.random()<=total_weight:
                    d_optimizer.zero_grad() 
                    real_out = D(real_img)
                    real_out.backward(mone)
                    x=np.arange(fake0.size(0))
                    idx=random.sample(list(x),BATCH_SIZE)
                    fake_img=fake0[idx,].to(device)
                    fake_out = D(fake_img)
                    fake_out.backward(one)
                    gradient_penalty = compute_gradient_penalty(D,real_img.data,fake_img.data)
                    gradient_penalty.backward(one)
                    WD=real_out.mean()-fake_out.mean()
                    d_loss = -real_out+fake_out+gradient_penalty
                    d_optimizer.step()

            if (epoch+1)<epoch_recording and (i+1)%20==0:
                z = torch.randn(num_img,z_dimension).to(device)
                fake_img = G(z).cpu()
                fake0=torch.cat([fake0, fake_img]).detach()
            for ii in range(1): 
                g_optimizer.zero_grad() 
                z = torch.randn(num_img,z_dimension).to(device)
                fake_img = G(z)
                fake_out = D(fake_img)
                fake_out.backward(mone)
                g_loss =  -fake_out
                g_optimizer.step()
            
            if epoch==0:
                real_images = (real_img.to(device).data)
                torchvision.utils.save_image(real_images, './save_dir/real_images.jpg')

            if (epoch+1)%20==0:
                fake_images =(fake_img.to(device))
                torchvision.utils.save_image(fake_images, './save_dir/fake_images-{}.jpg'.format(epoch + 1))

    torch.save(G, os.path.join(outdir, 'generator.pth'))
    torch.save(D, os.path.join(outdir, 'discriminator.pth'))

  

if __name__ == '__main__':  
 
    D = WGAN_D().to(device) 
    D.apply(weights_init)

    
    G = WGAN_G().to(device)  
    G.apply(weights_init)
   
    train(D, G, './save_dir', Z_DIM,num_epochs) 
   