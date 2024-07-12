import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange 


class LayerNorm2d(nn.Module): 
    def __init__(self, n_feats):
        super(LayerNorm2d, self).__init__() 
        
        self.norm = nn.LayerNorm(n_feats) 
    
    def forward(self, x): 
        
        x= rearrange(x, 'b c h w -> b h w c') 
        x = self.norm(x)
        x= rearrange(x, 'b h w c -> b c h w') 
        return x 
    

class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()


        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
    )

    def forward(self, x):
        # b, c, _, _ ,_= x.size()
        y = self.avg_pool(x)

        out = self.se(y)*x
        return out 

class Routing_Module(nn.Module):
    def __init__(self, global_dim, hidden_dim, num_expters):
        super(Routing_Module, self).__init__() 
        
        self.conv1 = nn.Conv2d(in_channels=global_dim+hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 

        self.gelu = nn.GELU() 
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=num_expters, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.relu = nn.ReLU()


    def forward(self, x, h): 
        out = self.conv1(torch.cat([x, h], dim=1))
        hidden = self.gelu(out) 

        out = self.conv3(hidden)
        logit = self.relu(out)

        return logit, hidden 




class Attention_Expert(nn.Module):
    def __init__(self, n_feats, ch_exp_f = 3, reduction=16, num_heads=16, bias=False):
        super(Attention_Expert, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(n_feats, n_feats*3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, groups=n_feats, bias=bias)
        self.project_out = nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=bias)
        


    def forward(self, x): 

        
        b,c,h,w = x.shape 

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) 
        
        out = self.dwconv(out)

        out = self.project_out(out) 

        return out 


class Attention_Bank(nn.Module):
    def __init__(self, n_feats, ch_exp_list = [1, 2, 3], reduction=16):
        super(Attention_Bank, self).__init__()
        self.num_expters = len(ch_exp_list)
        
        
        
        self.att_experts = nn.ModuleList([Attention_Expert(n_feats, ch_exp_f=ch_exp_list[i], reduction=reduction) for i in range(self.num_expters)]) 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.routing = Routing_Module(global_dim=n_feats, hidden_dim=16, num_expters=self.num_expters)


    def forward(self, x, hidden): 

        b, c, h, w = x.size() 
        x_global = self.avg_pool(x) 
        logit, hidden_new = self.routing(x_global, hidden) 
        logit = logit.unsqueeze(-1)

        for i, att_layer in enumerate(self.att_experts):
            if i == 0:
                out_matrix = att_layer(x).view(b, 1, c, h, w)
            else:
                out_matrix = torch.cat((out_matrix, att_layer(x).view(b,1, c, h, w)), dim=1)
        out = torch.sum(out_matrix*logit, dim=1)

        return  out, hidden_new, logit


class MLP_Expert(nn.Module):

    def __init__(self, in_feat, h_feat=None, out_feat=None):
        super().__init__()
        
        self.fc1 = nn.Conv2d(in_channels=in_feat, out_channels=h_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels=h_feat, out_channels=out_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class MLP_Bank(nn.Module):
    def __init__(self, n_feats, mlp_ratio_list = [1, 2, 4]):
        super(MLP_Bank, self).__init__()
        self.num_expters = len(mlp_ratio_list)
        
        
        
        self.mpl_experts = nn.ModuleList([MLP_Expert(in_feat=n_feats, h_feat=int(mlp_ratio_list[i]*n_feats), out_feat=n_feats) for i in range(self.num_expters)]) 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.routing = Routing_Module(global_dim=n_feats, hidden_dim=16, num_expters=self.num_expters)


    def forward(self, x, hidden): 

        b, c, h, w = x.size() 
        x_global = self.avg_pool(x) 
        logit, hidden_new = self.routing(x_global, hidden) 
        logit = logit.unsqueeze(-1)

        for i, mlp_layer in enumerate(self.mpl_experts):
            if i == 0:
                out_matrix = mlp_layer(x).view(b, 1, c, h, w)
            else:
                out_matrix = torch.cat((out_matrix, mlp_layer(x).view(b,1, c, h, w)), dim=1)
        out = torch.sum(out_matrix*logit, dim=1)

        return  out, hidden_new, logit


class Transformer_Block(nn.Module):
    
    def __init__(self, n_feats, 
                 ch_exp_list=[1,2,3],
                 reduction=16,
                 mlp_ratio_list=[1,2,4],

                 ): 
        super(Transformer_Block, self).__init__() 
        
        self.norm1 = LayerNorm2d(n_feats) 

        self.attention = Attention_Bank(n_feats, ch_exp_list = ch_exp_list, reduction=reduction)
        
        
        self.norm2 = LayerNorm2d(n_feats) 
        
        self.mlp = MLP_Bank(n_feats, mlp_ratio_list = mlp_ratio_list)

        

        self.beta = nn.Parameter(torch.ones((1, n_feats, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, n_feats, 1, 1)), requires_grad=True)

        
    def forward(self, x, hidden): 
        
        skip = x.clone()
        out = self.norm1(x)
        out, hidden, logit1 = self.attention(out, hidden) 
        out = out+self.beta*skip 
        
        skip = out.clone()
        
        out = self.norm2(out)
        out, hidden, logit2 = self.mlp(out, hidden) 
        
        out = out + self.gamma*skip
        
        return out, hidden, torch.cat([logit1, logit2], dim=-1)

class Generator(nn.Module):
    def __init__(self, 
                res_num=6,
                n_feats=64, 
                ch_exp_list=[1,1,1],
                reduction=8,
                mlp_ratio_list=[2, 2, 2],
                loss_fun=None
                 ):
        super(Generator, self).__init__() 
        
        self.head_conv = nn.Conv2d(1, n_feats, 3, padding=1, stride=1)
        

        
        blocks = []
        
        for i in range(res_num):
            blocks.append(Transformer_Block(
                              n_feats=n_feats, 
                              ch_exp_list=ch_exp_list,
                              reduction=reduction,
                              mlp_ratio_list=mlp_ratio_list,

                              ))

        self.body = nn.Sequential(*blocks) 
        
        self.tail_conv = nn.Conv2d(n_feats, 1, 3, padding=1, stride=1) 
        
        self.loss_fun = loss_fun 
        
        self.hidden = nn.Parameter(torch.ones((1, 16, 1, 1)), requires_grad=True)



    def forward(self, x, label=None):
        
        bs = x.shape[0] 
        # import pdb 
        # pdb.set_trace()
        hidden = self.hidden.repeat(bs, 1, 1, 1) 
        

        out = self.head_conv(x) 
        


        for i, blk in enumerate(self.body): 

            out, hidden, _ = blk(out, hidden) 
                

        
        out = self.tail_conv(out) + x
        
        if label is None:
            return out
        else:
            c_loss = self.loss_fun(out, label) 

            return c_loss 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

# model = MedRWKV()
# num = count_parameters(model) 
# print(num/1e6)
if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='7' 
    # x=torch.zeros((1,3,513,513)).type(torch.FloatTensor).cuda() 
    
    import time 
    
    # # y = mapping(x)
    # G=IPT() 
    # G.cuda()
    # with torch.no_grad():
    #     y=G(x) 
    # # print(time.time()-since) 
    from thop import profile, clever_format
    
    x=torch.zeros((1,1,128, 128)).type(torch.FloatTensor).cuda() 
    model = Generator() 
    # print(model)
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))
