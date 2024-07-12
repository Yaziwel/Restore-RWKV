import os 
os.environ['CUDA_VISIBLE_DEVICES']='7'  
from model.Restore_RWKV import Restore_RWKV
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Test_Data
import numpy as np
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm
import pdb 
import pandas as pd

transformData = transformData()
io=dataIO() 


data_root = "/home/data/zhiwen/dataset/All-in-One/" ### Path to place data
modality_list = ["PET", "CT", "MRI"] 
save_dir = "experiment/Restore_RWKV"


Generator = Restore_RWKV() 
Generator.cuda() 


Generator.load_state_dict(torch.load(os.path.join(save_dir, "Model","Generator_best.pth"))) 
Generator.eval()
for modality_name in modality_list:
    test_loader = DataLoader(Test_Data(root_dir=data_root, modality_list = [modality_name], target_folder="test"), batch_size=1, shuffle=False, num_workers=4) 
    psnr_list=[]
    ssim_list=[]
    rmse_list=[]
    name_list=[]
    for counter, data in enumerate(tqdm(test_loader)):
        v_in_pic, v_label_pic, modality, file_name = data 
        modality = modality[0] 
        file_name = file_name[0] 
    
        v_in_pic = v_in_pic.type(torch.FloatTensor).cuda() 
        v_label_pic = v_label_pic.type(torch.FloatTensor) 
        

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
        
        psnr_list.append(oneEval[0])
        ssim_list.append(oneEval[1])
        rmse_list.append(oneEval[2])
        name_list.append(file_name)

        io.save(gen_img.clone().numpy().squeeze(), os.path.join(save_dir, "test_result", modality, "{}.nii".format(file_name)))  
    
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list) 
    rmse_list = np.array(rmse_list)
    name_list = np.array(name_list)
    c_psnr = psnr_list.mean()
    c_ssim = ssim_list.mean()
    c_rmse = rmse_list.mean()
    print(" ^^^Final Test  {}   psnr:{:.6}, ssim:{:.6}, rmse:{:.6} ".format(modality_name, c_psnr,c_ssim, c_rmse))
    result_dict={
        "NAME":name_list,
        "PSNR":psnr_list,
        "SSIM":ssim_list, 
        "RMSE":rmse_list,
        }
    result=pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })
    result.to_csv(os.path.join(save_dir, "test_result", "{}_result.csv".format(modality_name))) 
