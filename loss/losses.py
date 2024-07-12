import torch.nn.functional as F
from torch import nn
import torch
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from torch.autograd import Variable
def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3, 3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3, 1, 3, 3, 3))

    sobelFilter[0, 0, 0, :, :] = num_1
    sobelFilter[0, 0, 1, :, :] = num_2
    sobelFilter[0, 0, 2, :, :] = num_3
    sobelFilter[1, 0, :, 0, :] = num_1
    sobelFilter[1, 0, :, 1, :] = num_2
    sobelFilter[1, 0, :, 2, :] = num_3
    sobelFilter[2, 0, :, :, 0] = num_1
    sobelFilter[2, 0, :, :, 1] = num_2
    sobelFilter[2, 0, :, :, 2] = num_3

    return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))


def sobelLayer(input):
    pad = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), -1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding=0, groups=1)/4
    n, c, h, w, l = fake_sobel.size()
    fake = torch.norm(fake_sobel, 2, 1, True)/c*3
    fake_out = act(fake)*2-1
    return fake_out

class EdgeAwareLoss(_WeightedLoss):

    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        self.sobelLayer = sobelLayer
        self.baseloss = nn.L1Loss()

    def forward(self, input, target):
        sobelFake = self.sobelLayer(input)
        sobelReal = self.sobelLayer(target)
        return self.baseloss(sobelFake,sobelReal)

class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,input, target):
        value=torch.sqrt(torch.pow(input-target,2)+self.epsilon2)
        return torch.mean(value)


class Centeral_Difference_Loss(nn.Module): 
    def __init__(self):
        super().__init__() 
        

        self.criterion = nn.L1Loss()
        

    def overlap_expand3D(self, x, kernel_size=3, stride=1, padding=1):
        B, C, D, H, W = x.shape 
        num_D=int((D+2*padding-kernel_size)/stride+1) 
        num_H=int((H+2*padding-kernel_size)/stride+1) 
        num_W=int((W+2*padding-kernel_size)/stride+1) 
        
        # import pdb 
        # pdb.set_trace()
        
        x=F.pad(x, (padding, padding, padding, padding, padding, padding))
        x_patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)  ###(position, kernel_size, stride)
        out = x_patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(B, C, kernel_size*num_D, kernel_size*num_H, kernel_size*num_W) 
        
        return out 
    
    def forward(self, x, y): 
        
        x_expand = self.overlap_expand3D(x) 
        x_up = F.interpolate(x, scale_factor=3, mode="nearest") 
        x_diff = x_up-x_expand

        y_expand = self.overlap_expand3D(y) 
        y_up = F.interpolate(y, scale_factor=3, mode="nearest") 
        y_diff = y_up-y_expand
        
        return self.criterion(x_diff, y_diff)