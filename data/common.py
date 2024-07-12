import torch
import os
import SimpleITK as sitk 
import pickle
# import pydicom
import numpy as np
import datetime
import pandas as pd
import json 
import random

def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p)) 


class transformPET:
    def __init__(self, data_range, mode="linear"):
        self.r = data_range
        self.m = mode

    def cut(self,x):
        thresh=self.r
        x[x>thresh]=thresh
        x[x<=0]=0.0
        _, _, d,h,w=x.shape
        x = x[:, :, 4:d-4, 16:h-16, 16:w-16]
        return x 

    def clip(self,x):

        x[x>self.r]=self.r
        x[x<=0]=0.0
        return x 
    
    def get_error_mask(self, low, full, error_thresh=0.01):
        error = torch.abs(full-low) 
        mask = (error>error_thresh).type(torch.FloatTensor).to(full.device)
        return mask
    
    def normalize(self, img, cut=False):
        img = self.clip(img)
        if self.m == "exp":
            c = torch.log(torch.Tensor([self.r+1])).to(img.device)
            img = torch.log(1+img)/c
        else:
            img = img/self.r
        
        if cut:
            img = self.cut(img)
        return img
    def denormalize(self, img):
       	if self.m=='exp':
       		c = torch.log(torch.Tensor([self.r+1])).to(img.device)
       		img = torch.exp(c*img)-1
       	else:
       		img*=self.r
       	img = self.clip(img)
        return img 



class transformData: 
    
    '''
    all-in-one medical image data
    '''
    
    def __init__(self, data_range=None):
        self.r = data_range 
        self.data_range = { 
            "CT":[-1024.0, 3072.0], 
            "PET":[0.0, 20.0], 
            "MRI":[0.0, 4095.0],
            } 
        
        
        '''
        Abdomen CT image is truncated to [-160, 240] 
        reference: https://github.com/SSinyu/WGAN-VGG
        '''
        
        self.test_trucate_data_range = { 
            "CT":[-160.0, 240.0], 
            "PET":[0.0, 20.0], 
            "MRI":[0.0, 4095.0],
            }


    def truncate(self,img, d_min, d_max):
        img[img>d_max]=d_max
        img[img<d_min]=d_min
        return img 
    
    def truncate_test(self, img, modality): 
        d_min, d_max = self.test_trucate_data_range[modality]
        img[img>d_max]=d_max
        img[img<d_min]=d_min
        return img
    
    def normalize(self, img, modality): 
        d_min, d_max = self.data_range[modality] 
        img = self.truncate(img, d_min, d_max) 
        img = (img - d_min)/(d_max - d_min)
        
        return img
    def denormalize(self, img, modality): 
        d_min, d_max = self.data_range[modality] 
        img = img*(d_max - d_min) + d_min 
        img = self.truncate(img, d_min, d_max)
        return img

    def random_crop(self, tensor, patch_size):
        """
        从给定的图像张量中随机裁剪大小为patch_size的patch。
    
        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
        patch_size: 裁剪patch的大小，格式为(H, W)。
    
        返回:
        裁剪后的patch张量。
        """
        B, C, H, W = tensor.shape
        patch_h, patch_w = patch_size
    
        # 确保裁剪尺寸不大于原图像尺寸
        if patch_h > H or patch_w > W:
            raise ValueError("裁剪尺寸应小于原始图像尺寸")
    
        # 随机选择裁剪的起始点
        top = random.randint(0, H - patch_h)
        left = random.randint(0, W - patch_w)
    
        # 裁剪patch
        patches = tensor[:, :, top:top + patch_h, left:left + patch_w]
        return patches


    def random_rotate_flip(self, tensor):
        """
        对形状为[B, C, H, W]的图像张量执行随机旋转或翻转。
    
        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
    
        返回:
        经过随机旋转或翻转的图像张量。
        """
        B, C, H, W = tensor.shape
        processed = torch.empty_like(tensor)
    
        for i in range(B):
            img = tensor[i]
            operation = torch.randint(0, 6, (1,)).item()
    
            if operation == 1:
                # 水平翻转
                img = torch.flip(img, [2])
            elif operation == 2:
                # 垂直翻转
                img = torch.flip(img, [1])
            elif operation == 3:
                # 旋转90度
                img = img.transpose(1, 2).flip(2)
            elif operation == 4:
                # 旋转180度
                img = img.flip(1).flip(2)
            elif operation == 5:
                # 旋转270度
                img = img.transpose(1, 2).flip(1)
    
            # 不做改变的情况下，operation == 0
            processed[i] = img
    
        return processed


    def _add_gaussian_noise(self, clean_patch, sigma):
        # 将sigma从[0,255]范围转换到[0,1]范围
        sigma = sigma / 255.0
        noise = torch.randn_like(clean_patch)
        noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 1)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patches, degrade_type=None):
        if degrade_type is None:
            degrade_type = random.randint(0, 2)

        B, C, H, W = clean_patches.shape
        degraded_patches = torch.empty_like(clean_patches)

        for i in range(B):
            degraded_patches[i], _ = self._degrade_by_type(clean_patches[i], degrade_type)

        return degraded_patches




@torch.no_grad()
def synthesisOneAxial(model, img, kernel_size=64, stride=32, crop_size = 3):
    model.eval()
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride==D:
            break
        elif idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ######
        G_patches = model(patches)
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

@torch.no_grad()
def merge_patch(model, img, kernel_size=64, stride=32,crop_size = 3):
    model.eval()
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride==D:
            break
        elif idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ###### 
        G_patches = patches.clone().detach() 
        for i in range(len(patches)):
            G_patches[i] = model(patches[[i],:,:,:,:])
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

class dataIO:
    def __init__(self):
        self.reader = {
            '.img':self.load_itk,
            '.gz':self.load_itk, 
            '.nii':self.load_itk,
            '.bin':self.load_bin, 
            '.txt':self.load_txt, 
            '.json':self.load_json
            
            }
        self.writer = {
            '.img':self.save_itk, 
            '.gz':self.save_itk, 
            '.nii':self.save_itk,
            '.bin':self.save_bin,
            '.csv':self.save_csv,
            '.txt':self.save_txt, 
            '.txt':self.save_json
            }
    
    
    def save_itk(self, data, path, use_int=False): 
        if use_int: 
            data = np.around(data)
        sitk.WriteImage(sitk.GetImageFromArray(data), path) 

    def save_bin(self, data, path, use_int=False): 
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
    def load_itk(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
        
    def load_bin(self,path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data 
    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read() 
        return data



    def save_json(self, data, path):
        with open(path, "w", encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 

    def load_json(self, path):
        with open(path, encoding='utf8') as f:
            data = json.load(f) 
        return data 

    def save_csv(self, data_dict, path):
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)
    
    def save_txt(self, s, path):
        with open(path,'w') as f:
            f.write(s) 
            

        
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex
    
    def load(self, path):
        ex = self.getFileEX(path)
        return self.reader[ex](path)
    def save(self, data, path): 
        mkdir(path, is_file=True)
        ex = self.getFileEX(path)
        return self.writer[ex](data, path)




