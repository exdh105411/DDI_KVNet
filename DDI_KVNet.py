import os
from collections import OrderedDict
from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
from time import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import platform
import glob
from argparse import ArgumentParser
import random
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, datasets
import models.basicblock as B
from models.network_unet import UNetRes as net
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
import cv2
##################################################################################### 
class FFT_image(nn.Module):
    def __init__(self):
        super(FFT_image, self).__init__()

    def forward(self, x, mask=None):
        num=x.shape[1]//2
        x_real = x[:, 0:num, :, :]
        x_imag = x[:, num:, :, :]
        x_complex = torch.complex(x_real, x_imag)
        fftz = torch.fft.fft2(x_complex)
        x_fft=fftz
        if mask!=None:
            x_fft = fftz * mask
        x_fft = torch.cat([x_fft.real,x_fft.imag],1)
        return x_fft
 
class iFFT_image(nn.Module):
    def __init__(self):
        super(iFFT_image, self).__init__()

    def forward(self, x,only_real=True):
        num=x.shape[1]//2
        x_real = x[:, 0:num, :, :]
        x_imag = x[:, num:, :, :]
        x_complex = torch.complex(x_real, x_imag)
        z_hat = torch.fft.ifft2(x_complex)
        if only_real :
            return z_hat.real
        x_ifft=torch.cat([z_hat.real,z_hat.imag],dim=1)
        #print(x_ifft.shape)
        return x_ifft
#####################################################################################   
class VNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=2, nc=[32, 64, 128, 256], nb=1, act_mode='R'):
        super(VNet,self).__init__()
        self.D1=B.sequential(B.conv(in_nc, nc[0], mode='C'+act_mode[-1],bias=False),*[B.conv(nc[0], nc[0], mode='C'+act_mode,bias=False) for _ in range(nb)])
        self.D2=B.sequential(B.conv(nc[0], nc[1], mode='C'+act_mode[-1],bias=False),*[B.conv(nc[1], nc[1], mode='C'+act_mode,bias=False) for _ in range(nb)])
        self.D3=B.sequential(B.conv(nc[1], nc[2], mode='C'+act_mode[-1],bias=False),*[B.conv(nc[2], nc[2], mode='C'+act_mode,bias=False) for _ in range(nb)])
        self.E1=B.sequential(B.CALayer(nc[0],8,True),*[B.conv(nc[0], nc[0], mode='C'+act_mode,bias=False) for _ in range(nb+1)],B.conv(nc[0], out_nc, mode='C',kernel_size=1,stride=1,padding=0,bias=False))
        self.E2=B.sequential(B.CALayer(nc[1],8,True),B.conv(nc[1], nc[0], mode='C'+act_mode,bias=False),*[B.conv(nc[0], nc[0], mode='C'+act_mode,bias=False) for _ in range(nb)])
        self.E3=B.sequential(B.CALayer(nc[2],8,True),B.conv(nc[2], nc[1], mode='C'+act_mode,bias=False),*[B.conv(nc[1], nc[1], mode='C'+act_mode,bias=False) for _ in range(nb)])
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.up1=nn.ConvTranspose2d(in_channels=nc[2], out_channels=nc[2], kernel_size=2, stride=2, padding=0, bias=False)
        self.up2=nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.up3=nn.ConvTranspose2d(in_channels=nc[0], out_channels=nc[0], kernel_size=2, stride=2, padding=0, bias=False)
        self.bn=B.sequential(B.conv(nc[2],nc[3], mode='C'+act_mode[-1],bias=False),B.conv(nc[3],nc[2], mode='C',bias=False))
  
    def forward(self,x0):
        x1=self.D1(x0)
        x1_d=self.max_pool(x1)
        x2=self.D2(x1_d)
        x2_d=self.max_pool(x2)
        x3=self.D3(x2_d)
        x3_d=self.max_pool(x3)

        x=self.bn(x3_d)

        x=self.up1(x+x3_d)
        x=self.E3(x+x3)
        x=self.up2(x+x2_d)
        x=self.E2(x+x2)
        x=self.up3(x+x1_d)
        x=self.E1(x+x1)
        return x
    
##################################################################################### 
class FALayer(nn.Module):
    '''frequency attention layer'''
    def __init__(self,in_nc=64):
        super(FALayer,self).__init__()
        self.sig = nn.Sigmoid()
        self.cs_weight = nn.Conv2d(in_nc, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self,inputs):
        cs_x=self.cs_weight(inputs)
        cs_x=self.sig(cs_x)
        outputs=cs_x*inputs
        return outputs
    
class KNet_attention(nn.Module):
    '''the attention layer in KNet'''
    def __init__(self, in_nc=64,reduction=16):
        super(KNet_attention,self).__init__()
        self.channel_atten=B.CALayer(in_nc,reduction)
        self.frq_atten=FALayer(in_nc)
    def forward(self,inputs):
        x=torch.abs(inputs)
        x_frq=self.frq_atten(x)
        x_ch=self.channel_atten(x)
        outputs=torch.max(x_frq,x_ch)
        return outputs

class KNet_downLayer(nn.Module):
    '''Cross domain down sample'''
    def __init__(self,in_nc=64,out_nc=32,nb=1,act_mode='R'):
        super(KNet_downLayer,self).__init__()
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv=B.sequential(B.conv(in_nc, out_nc, mode='C'+act_mode[-1],bias=False),*[B.conv(out_nc,out_nc, mode='C'+act_mode,bias=False) for _ in range(nb)],KNet_attention(out_nc,2))
        self.fft=FFT_image()
        self.ifft=iFFT_image()
    def forward(self,x0):
        x_ifft=self.ifft(x0,only_real=False)
        x_d=self.max_pool(x_ifft)
        x_fft=self.fft(x_d)
        x=self.conv(x_fft)
        return x

class KNet_upLayer(nn.Module):
    '''Cross domain up sample'''
    def __init__(self, in_nc=64,out_nc=32,nb=1,act_mode='R'):
        super(KNet_upLayer,self).__init__()
        self.up=nn.ConvTranspose2d(in_channels=in_nc, out_channels=out_nc, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv=B.sequential(B.conv(in_nc, out_nc, mode='C'+act_mode[-1],bias=False),*[B.conv(out_nc, out_nc, mode='C'+act_mode,bias=False) for _ in range(nb)],KNet_attention(out_nc,2))
        self.fft=FFT_image()
        self.ifft=iFFT_image()
    def forward(self,x0,skip_con):
        x_ifft=self.ifft(x0,only_real=False)
        x_up=self.up(x_ifft)
        x_fft=self.fft(x_up)
        x_cat=torch.cat([x_fft,skip_con],dim=1)
        x=self.conv(x_cat)
        return x

class KNet(nn.Module):
    def __init__(self,in_nc=2, out_nc=2, nc=[8, 16, 32, 64], nb=1, act_mode='R'):
        super(KNet,self).__init__()
        self.head=B.sequential(B.conv(in_nc, nc[0], mode='C'+act_mode[-1],bias=False),*[B.conv(nc[0], nc[0], mode='C'+act_mode,bias=False) for _ in range(nb)],KNet_attention(nc[0],2))
        self.D1=KNet_downLayer(nc[0],nc[1],nb,act_mode)
        self.D2=KNet_downLayer(nc[1],nc[2],nb,act_mode)
        self.body=KNet_downLayer(nc[2],nc[3],nb,act_mode)
        self.U1=KNet_upLayer(nc[3],nc[2],nb,act_mode)
        self.U2=KNet_upLayer(nc[2],nc[1],nb,act_mode)
        self.U3=KNet_upLayer(nc[1],nc[0],nb,act_mode)
        self.tail=B.conv(nc[0],2,kernel_size=1,stride=1,padding=0,bias=False,mode='C')
    
    def forward(self,x0):
        x_h=self.head(x0)
        x1=self.D1(x_h)
        x2=self.D2(x1)
        x_b=self.body(x2)
        x=self.U1(x_b,x2)
        x=self.U2(x,x1)
        x=self.U3(x,x_h)
        x=self.tail(x)
        return x
    
##################################################################################### 

class KDC(nn.Module):
    '''K space Data consistency'''
    def __init__(self):
        super(KDC,self).__init__()
    def forward(self,inputs,y0,mask,alpha):
        outputs=inputs+(alpha/(1.0+alpha))*(y0-inputs*mask)
        return outputs
    
class VDC(nn.Module):
    '''Image Data consistency'''
    def __init__(self):
        super(VDC,self).__init__()
        self.fft=FFT_image()
        self.ifft=iFFT_image()
    def forward(self,inputs,y0,mask,alpha):
        x_fft=self.fft(inputs)
        x=x_fft+(alpha/(1.0+alpha))*(y0-x_fft*mask)
        outputs=self.ifft(x,only_real=False)
        return outputs
    
##################################################################################### 

class DDI(nn.Module):
    '''double domain interaction'''
    def __init__(self):
        super(DDI,self).__init__()
        self.fft=FFT_image()
        self.ifft=iFFT_image()
        self.conv_k=B.conv(in_channels=2,out_channels= 2, mode='C',bias=False)
        self.conv_i=B.conv(in_channels=2,out_channels=2,mode='C',bias=False)
    
    def forward(self,in_k,in_v):
        in_v_fft=self.fft(in_v)
        output_k=in_k+self.conv_k(in_v_fft)

        in_k_ifft=self.ifft(in_k,only_real=False)
        output_v=in_v+self.conv_i(in_k_ifft)

        return output_k,output_v


class condition_network(nn.Module):
    '''Condition module CM'''
    def __init__(self,in_nc,out_nc):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(in_nc, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, out_nc, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x
    
##################################################################################### 

class DDI_KVNet(nn.Module):
    '''double domain interaction KVNet'''
    def __init__(self):
        super(DDI_KVNet,self).__init__()
        self.fft=FFT_image()
        self.ifft=iFFT_image()
        self.DDI=DDI()
        self.KDC=KDC()
        self.VDC=VDC()
        self.VNet=VNet(in_nc=2, out_nc=2, nc=[32, 64, 128, 256], nb=1, act_mode='R')
        self.KNet=KNet(in_nc=2, out_nc=2, nc=[8, 16, 32, 64], nb=1, act_mode='R')
        self.condition = condition_network(2,5)
    def forward(self,xk,yk,y0,mask,denoiser,cs_ratio,n_step,device):
        cond=torch.tensor([[cs_ratio/100.,n_step]]).type(torch.FloatTensor).to(device)
        [alpha1,alpha2,alpha3,alpha4,sigma]=self.condition(cond)[0]
        
        y_k1=self.KNet(yk)+yk
        x_k1=self.VNet(xk)+xk

        y_k1=self.KDC(y_k1,y0,mask,alpha1)
        x_k1=self.VDC(x_k1,y0,mask,alpha2)
        y_k1,x_k1=self.DDI(y_k1,x_k1)
        x_k1=x_k1[:,0:1,:,:]
        noise_level_map=sigma.repeat(x_k1.shape[0], 1, x_k1.shape[2], x_k1.shape[3])
        dn_in=torch.cat([x_k1,noise_level_map],dim=1)
        x_k1=denoiser(dn_in)
        zero=torch.zeros_like(x_k1).to(device)
        x_k1=torch.cat([x_k1,zero],dim=1)
        y_out=y_k1+alpha3*(y_k1-yk)
        x_out=x_k1+alpha4*(x_k1-xk)

        return x_out,y_out
    
##################################################################################### 

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
    
#####################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#denoiser
denoiser = net(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
denoiser.load_state_dict(torch.load('model_zoo/drunet_150.pth'))
print("Loaded pre-trained denoiser model.")
denoiser.eval()
for param in denoiser.parameters():
    param.requires_grad = False
denoiser = denoiser.to(device)

#load mask
rand_num = 1
matrix_dir='Cartesian_untrained'
if matrix_dir == '2D-random':
    train_cs_ratio_set = [5, 10, 20, 30, 40]
elif matrix_dir =='Cartesian_untrained':
    train_cs_ratio_set = [15, 25, 35, 45]
else:
    train_cs_ratio_set = [10, 20, 30, 40, 50]
Phi_all = {}
Phi = {}
for cs_ratio in train_cs_ratio_set:
    Phi_data_Name = './sampling_matrix/%s/mask_%d.mat' % (matrix_dir, cs_ratio)
    Phi_data = sio.loadmat(Phi_data_Name)
    mask_matrix = Phi_data['mask_matrix']
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), mask_matrix.shape[0], mask_matrix.shape[1]))

    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = mask_matrix[:, :]

    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)

#model
model = DDI_KVNet()
model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
fft=FFT_image()
ifft=iFFT_image()
loss_l1=nn.L1Loss()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

#dataset
batch_size = 8
data_dir='data'
Training_data_Name = 'Training_BrainImages_256x256_100.mat'
Training_data = sio.loadmat('./%s/%s' % (data_dir, Training_data_Name))
Training_labels = Training_data['labels']
nrtrain = Training_labels.shape[0]   # number of training image
dataset =RandomDataset(Training_labels, nrtrain)
print('Train data shape=',Training_labels.shape)
# transform=transforms.Compose([transforms.CenterCrop(256),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
# dataset=datasets.ImageFolder(root='dtd/images',transform=transform)
if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=6,
                             shuffle=True,pin_memory=True)

run_mode='test'
start_epoch=0
end_epoch=170
save_interval=10

#train
if run_mode=='train':
    if start_epoch>0:
        model.module.load_state_dict(torch.load('DDI_KVNet_model/NO_1_net_epochs_%d.pkl' % (start_epoch)))
    start_time=time()
    for epoch_i in range(start_epoch+1, end_epoch+1):
        model = model.train()
        step = 0
        max_iter=5
        if 20<epoch_i<=50:
            max_iter=10
        elif epoch_i>50:
            max_iter=20
        elif epoch_i>100:
            max_iter=30
        for data in rand_loader:
            time1=time()
            step = step+1
            
            #batch_x = data[0]
            batch_x=data
            batch_x = torch.unsqueeze(batch_x, 1).cpu().data.numpy()
            batch_x = torch.from_numpy(batch_x)
            zero=torch.zeros_like(batch_x)
            x=torch.cat([batch_x,zero],dim=1).to(device)

            rand_Phi_index = np.random.randint(rand_num * 1)
            rand_cs_ratio = np.random.choice(train_cs_ratio_set)
            #rand_cs_ratio = 10
            mask = Phi[rand_cs_ratio][rand_Phi_index]
            mask = torch.unsqueeze(mask, 0)
            mask = mask.to(device)

            y0 = fft(x, mask)
            xk=ifft(y0,only_real=False)
            yk=y0
            n_step=random.randint(1,max_iter)
            model=model.eval()
            with torch.no_grad():
                for i in range(1,n_step):
                    xk,yk=model(xk,yk,y0,mask,denoiser,rand_cs_ratio,i,device)
                    #a=ssim(xk[:,0:1,:,:].clamp(0,1),x[:,0:1,:,:],data_range=1,size_average=True)
                    #print(a)
            model=model.train()
            x_out,y_out = model(xk,yk,y0,mask,denoiser,rand_cs_ratio,n_step,device)
            time2=time()
            # Compute and print loss
          
            loss_ssim=1-ssim(x_out[:,0:1,:,:].clamp(0,1),x[:,0:1,:,:],data_range=1,size_average=True)
            l1_loss=loss_l1(x_out[:,0:1,:,:],x[:,0:1,:,:])
            loss_all=l1_loss
            if epoch_i>25:
                loss_all=0.16*l1_loss+0.84*loss_ssim
            #print(loss_ssim.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad(set_to_none=True)
            loss_all.backward()
            time3=time()
            optimizer.step()
            time4=time()
            #print("Time for forward pass: {:.4f} seconds, Time for backward pass: {:.4f} seconds, Time for optimizer step: {:.4f} seconds".format(time2-time1, time3-time2, time4-time3))
       
            # step %100==0
            if step % 100 == 0:
                step_time=time()-start_time
                output_data = "time:%.2f [%02d/%02d] Step:%.0f | CS ratio:%.0f | n_step:%d | Total Loss: %.6f | SSIM Loss: %.4f" % \
                            (step_time, epoch_i, end_epoch, step,rand_cs_ratio,n_step,loss_all.item(),loss_ssim.item())
                print(output_data)

            # Load pre-trained model with epoch number
        
        
        # save model in every epoch
        if epoch_i % save_interval ==0:
            torch.save(model.module.state_dict(), "DDI_KVNet_model/NO_1_net_epochs_%d.pkl" % ( epoch_i))  # save only the parameters

elif run_mode=='test':
    model.module.load_state_dict(torch.load('DDI_KVNet_model/NO_1_net_epochs_%d.pkl' % (end_epoch)))
    model=model.eval()
    PSNRL=[]
    SSIML=[]
    for cs_ratio in train_cs_ratio_set:
        psnr_step=[]
        ssim_step=[]
        mask = Phi[cs_ratio][0]
        mask = torch.unsqueeze(mask, 0)
        mask = mask.to(device)
        print(f"Testing with ratio={cs_ratio}...")
        for n in range(1,51):
            img=util.imread_uint(f'data/BrainImages_test/brain_test_{n:02d}.png', n_channels=1).squeeze()
            batch_x=util.uint2tensor4(img)
            zero=torch.zeros_like(batch_x)
            x=torch.cat([batch_x,zero],dim=1).to(device)
            y0 = fft(x, mask)
            xk=ifft(y0,only_real=False)
            yk=y0
            
            
            #time1=time()
            with torch.no_grad():
                for i in range(1,15):
                    xk,yk=model(xk,yk,y0,mask,denoiser,cs_ratio,i,device)
                #x_m=fft_mask_forback(batch_x,expand_mask)
                # a=ssim(xk[:,0:1,:,:].clamp(0,1),x[:,0:1,:,:],data_range=1,size_average=True)
                # print(a)
                x_am=util.tensor2single(batch_x)*255.
                x_pre=util.tensor2single(xk[:,0:1,:,:])*255.
            
                psnr_=util.calculate_psnr(x_pre,x_am)
                ssim_=util.calculate_ssim(x_pre,x_am)
            
                psnr_step.append(psnr_)
                ssim_step.append(ssim_)

                # print(psnr_,ssim_)
                im_rec_rgb = np.clip(x_pre, 0, 255).astype(np.uint8)
                img_name = f"brain_test_{n:02d}"
                os.makedirs(f'result/Brain_test_DDI_KVNet_epoch_{end_epoch}/', exist_ok=True)
                img_dir =f'result/Brain_test_DDI_KVNet_epoch_{end_epoch}/' + img_name+ "_ratio_%d_PSNR_%.3f_SSIM_%.5f.png" % (cs_ratio,psnr_, ssim_)
                
                cv2.imwrite(img_dir, im_rec_rgb)
        PSNRL.append(np.mean(psnr_step))
        SSIML.append(np.mean(ssim_step))
    with open(f'DDI_KVNet_{end_epoch}.txt','a') as f:
        
        f.write('\n')
        
        f.writelines([str(x) + ',' for x in PSNRL])
        f.write('\n')
        f.writelines([str(x) + ',' for x in SSIML])
    #print(PSNRL,SSIML)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # x=[i for i in range(1,21)]
    # plt.plot(x, PSNRL, marker='o',c='blue')
    # plt.xlabel('n_step')
    # plt.ylabel('PSNR')
    # plt.title(f'{cs_ratio}% PSNR')
    

    # plt.subplot(1, 2, 2)
    # plt.plot(x, SSIML, marker='o',c='blue')
    # plt.xlabel('n_step')
    # plt.ylabel('SSIM')
    # plt.title(f'{cs_ratio}% SSIM')
    

    # #plt.tight_layout()
    # plt.show()