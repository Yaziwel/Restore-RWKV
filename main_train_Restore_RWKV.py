import os 
os.environ['CUDA_VISIBLE_DEVICES']='5' 
from model.Restore_RWKV import Restore_RWKV
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Train_Data, Test_Data, DataSampler
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tools import set_seeds, mkdir  


transformData = transformData()
io=dataIO() 
# set_seeds(3042)



def save_model(G_net_model, save_dir, optimizer_G=None, ex=""):
    save_path=os.path.join(save_dir, "Model")
    mkdir(save_path)
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex))
    torch.save(G_net_model.cpu().state_dict(), G_save_path)
    G_net_model.cuda()

    if optimizer_G is not None:
        opt_G_save_path = os.path.join(save_path,'Optimizer_G{}.pth'.format(ex))
        torch.save(optimizer_G.state_dict(), opt_G_save_path)

def build_train_sampler(modality_list, data_root, batch_size, shuffle=True):
    dataset = Train_Data(root_dir = data_root, modality_list = modality_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4) 
    sampler = DataSampler(dataloader)
    print("data length: \n", dataset.length) 
    
    return sampler



'''
Testing Code
'''


total_iteration = 3e5
val_iteration = 1e3
lr = 2e-4

batch_size = 4
eps=1e-8

psnr_max=0

save_dir = "experiment/Restore_RWKV" 
data_root = "/home/data/zhiwen/dataset/All-in-One/" 
# modality_list = ["PET", "CT", "MRI"] 
modality_list = ["MRI"] 

Generator = Restore_RWKV() 
Generator.cuda() 

train_sampler = build_train_sampler(modality_list, data_root, batch_size, shuffle=True) 
valid_loader = DataLoader(Test_Data(root_dir=data_root, use_num=32, modality_list=modality_list), batch_size=1, shuffle=False) 


optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08) 
lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-6)
criterion = nn.L1Loss().cuda()

running_loss = []
eval_metrics={
    "psnr":[],
    "ssim":[], 
    "rmse":[]
    }

pbar = tqdm(total=int(total_iteration))

print("################ Train ################")
for iteration in list(range(1, int(total_iteration)+1)):


    l_G=[]
    in_pic,  label_pic, class_label = next(train_sampler)
    
    in_pic = in_pic.type(torch.FloatTensor).cuda()
    label_pic = label_pic.type(torch.FloatTensor).cuda() 

    #################
    #     train G
    #################
    Generator.train()
    optimizer_G.zero_grad() 
    


    restored = Generator(in_pic) 
    loss_G = criterion(restored, label_pic) 
    
    
    loss_G.backward()
    optimizer_G.step()

    l_G.append(loss_G.item())
    torch.cuda.empty_cache() 
    lr_scheduler_G.step()
            
    

    
    if iteration % val_iteration == 0: 
        psnr=0
        ssim=0
        rmse=0
        Generator.eval() 
        for counter,data in enumerate(tqdm(valid_loader)):
            v_in_pic, v_label_pic, modality, file_name = data 
            modality = modality[0] 
            file_name = file_name[0] 

            v_in_pic = v_in_pic.type(torch.FloatTensor).cuda() 
            v_label_pic = v_label_pic.type(torch.FloatTensor)
            
            # pdb.set_trace()
            with torch.no_grad():
                gen_img = Generator(v_in_pic) 
            
            gen_img = transformData.denormalize(gen_img, modality).detach().cpu() 
            
            v_label_pic = transformData.denormalize(v_label_pic, modality) 
            
            
            
            '''
            truncation for test image 
            CT:[-160, 240]
            '''
            
            gen_img = transformData.truncate_test(gen_img, modality) 
            v_label_pic = transformData.truncate_test(v_label_pic, modality) 
            
            data_range = v_label_pic.max()-v_label_pic.min()
            oneEval = compute_measure(gen_img, v_label_pic, data_range = data_range) 
            
            psnr+=oneEval[0]
            ssim+=oneEval[1] 
            rmse+=oneEval[2] 

            io.save(gen_img.clone().numpy().squeeze(), os.path.join(save_dir, "Gimg", "{}_{}.nii".format(file_name, modality) ))
            
            torch.cuda.empty_cache()
        c_psnr=psnr/(counter+1)
        c_ssim=ssim/(counter+1) 
        c_rmse=rmse/(counter+1) 
        
        eval_metrics['psnr'].append(c_psnr)
        eval_metrics['ssim'].append(c_ssim)  
        eval_metrics['rmse'].append(c_rmse) 
    
        save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G=None, ex="_iteration_{}".format(iteration))
        if c_psnr>=psnr_max:
            psnr_max=c_psnr
            io.save("Best Iteration: {}, PSNR: {}, SSIM:{}, RMSE:{}".format(iteration, c_psnr, c_ssim, c_rmse),os.path.join(save_dir, "best.txt"))
            save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G = optimizer_G, ex="_best")
        io.save(
            {'eval_metrics':eval_metrics},
            os.path.join(save_dir, "evaluationLoss.bin")
            ) 


    pbar.set_description("loss_G:{:6}, psnr:{:6}".format(loss_G.item(), eval_metrics['psnr'][-1] if len(eval_metrics['psnr'])>0 else 0)) 
    pbar.update() 
