import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO, transformData
import glob

io=dataIO() 
transform = transformData()

class Train_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET"], patch_size=128):

        
        
        
        self.LQ_paths = [] 
        self.HQ_paths = [] 
        
        for modality in modality_list:
            
            tmp_paths = glob.glob(os.path.join(root_dir, modality, "train", "LQ", "*.bin")) 
            
            for p in tmp_paths:  
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))

        self.length = len(self.LQ_paths) 
        self.label_dict = {
            "PET": 0, 
            "CT": 1, 
            "MRI": 2
            } 
        self.patch_size = patch_size


    def __len__(self):
        return self.length 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx]) 
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace() 
        modality, _ = self.analyze_path(self.LQ_paths[idx]) 
        
        imgLQ = transform.normalize(imgLQ, modality)
        imgHQ = transform.normalize(imgHQ, modality)

        cat_pic = torch.cat([imgLQ, imgHQ], dim=0).unsqueeze(1)
        cat_pic = transform.random_crop(tensor = cat_pic, patch_size=[self.patch_size, self.patch_size]).squeeze(1)
        imgLQ, imgHQ = torch.chunk(cat_pic, 2, dim=0) 
        
        class_label = self.label_dict[modality]
        
        return imgLQ, imgHQ, class_label


class Test_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET", "CT", "MRI"], use_num = None, target_folder="validation"): 
        
        self.LQ_paths = [] 
        self.HQ_paths = [] 
        
        
        
        for modality in modality_list: 
            tmp_paths = glob.glob(os.path.join(root_dir, modality, target_folder, "LQ", "*.nii")) 
            
            use_num = len(tmp_paths) if use_num is None else use_num
            
            for num in range(use_num): 
                p = tmp_paths[num]
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))  

        self.length = len(self.LQ_paths) 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name
        

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx])
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        modality, file_name = self.analyze_path(self.LQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace()

        
        imgLQ = transform.normalize(imgLQ, modality) 
        imgHQ = transform.normalize(imgHQ, modality) 
        
        imgLQ = torch.from_numpy(imgLQ).unsqueeze(0) 
        imgHQ = torch.from_numpy(imgHQ).unsqueeze(0)

        return imgLQ, imgHQ, modality, file_name








class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # 如果 DataLoader 中的数据采样完了，重新 shuffle 数据

            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch

# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 

if __name__ == "__main__": 
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    
    data_root = "/home/data/zhiwen/dataset/All-in-One/" 
    modality_list = ["PET", "CT", "MRI"]
    train_loader_list = []


    
    dataset = {
        'train': Train_Data(root_dir = data_root, modality = "PET"), 
        # 'val': Test_Data(root_dir = data_root, modality_list = ["MRI"], use_num=4), 
        # 'test': Test_Data(root_dir=data_root, center_name="m660-1", use_num=-1),
        
        } 
    train_loader = DataLoader(dataset['train'], batch_size=1, shuffle=False) 
    
    print("length:", len(train_loader))
    
    # valid_loader = DataLoader(Val_Data(root=data_root, center_list=center_list), batch_size=1) 
    
    # test_loader = DataLoader(Test_Data(root=data_root, center_list=center_list), batch_size=1)
    
    

    for counter, data in enumerate(tqdm(train_loader)): 
        import pdb 
        pdb.set_trace()