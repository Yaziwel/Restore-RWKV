import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p)) 

def load_bin(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data 

# data = load_bin(r"evaluationLoss.bin")
# psnr = data['eval_metrics']['loss']
# plt.plot(psnr) 

# import torch.nn.functional as F

# # x1 = torch.tensor([[1,2,3],[4,5,6], [7,8,9]]).unsqueeze(0).unsqueeze(0)  
# # x2 = torch.tensor([[11,12,13],[14,15,16], [17,18,19]]).unsqueeze(0).unsqueeze(0)  
# # x3 = torch.tensor([[21,22,23],[24,25,26], [27,28,29]]).unsqueeze(0).unsqueeze(0)  
# # x = torch.cat([x1,x2,x3],dim=1).unsqueeze(0)

# def overlap_expand2D(x, kernel_size=3, stride=1, padding=1):
#     B, C, H, W = x.shape 
#     num_H=int((H+2*padding-kernel_size)/stride+1) 
#     num_W=int((W+2*padding-kernel_size)/stride+1) 
    
#     x=F.pad(x, (padding,padding,padding,padding))
#     x_patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)  ###(position, kernel_size, stride)
#     out = x_patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(1,C,kernel_size*num_H,kernel_size*num_W) 
    
#     return out


# def overlap_expand3D(x, kernel_size=3, stride=1, padding=1):
#     B, C, D, H, W = x.shape 
#     num_D=int((D+2*padding-kernel_size)/stride+1) 
#     num_H=int((H+2*padding-kernel_size)/stride+1) 
#     num_W=int((W+2*padding-kernel_size)/stride+1) 
    
#     # import pdb 
#     # pdb.set_trace()
    
#     x=F.pad(x, (padding, padding, padding, padding, padding, padding))
#     x_patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)  ###(position, kernel_size, stride)
#     out = x_patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(1, C, kernel_size*num_D, kernel_size*num_H, kernel_size*num_W) 
    
#     return out

# class logger:
#     def __init__(self, record_list = ["L_condition", "L_adv", "psnr", "ssim", "lpips"], best_mark="psnr"):
#         self.r = {}
#         for s in record_list:
#             self.r[s]=[]
#         self.tmp = {}
#         for s in record_list:
#             self.tmp[s]=[]
    
#     def make_tmp_record(self):
        
        
#     def plot_psnr(self, epoch):
#         axis = np.linspace(1, epoch, epoch)
#         for idx_data, d in enumerate(self.args.data_test):
#             label = 'SR on {}'.format(d)
#             fig = plt.figure()
#             plt.title(label)
#             for idx_scale, scale in enumerate(self.args.scale):
#                 plt.plot(
#                     axis,
#                     self.log[:, idx_data, idx_scale].numpy(),
#                     label='Scale {}'.format(scale)
#                 )
#             plt.legend()
#             plt.xlabel('Epochs')
#             plt.ylabel('PSNR')
#             plt.grid(True)
#             plt.savefig(self.get_path('test_{}.pdf'.format(d)))
#             plt.close(fig)
