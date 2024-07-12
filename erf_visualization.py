import os 
os.environ['CUDA_VISIBLE_DEVICES']='1' 
from model.Restore_RWKV import Restore_RWKV
from loss.losses import CharbonnierLoss
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Test_Data
import numpy as np
from timm.utils import AverageMeter
import torch
from torch import nn
from torch.utils.data import DataLoader 

transformData = transformData()
io=dataIO() 


if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24 
    sns_text_size = 4
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white") 
    sns.set(font_scale=sns_text_size)
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False


def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None, cbar=False):
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=0, annot=False, ax=ax, cbar=cbar, annot_kws={"size": 24}, fmt='.2f') 
        if cbar: 
            ax.collections[0].set_clim(0,1) 
        plt.savefig(save_path)

    def analyze_erf(args):
        data = args.source
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1] for the comparability among models
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args) 
    
# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module = None, 
                  num_images=1000, 
                  data_path="/home/data/zhiwen/dataset/All-in-One/" , 
                  # save_path=f"experiment/MedRWKV_Q_Shift_Re_WKV", 
                  modality_name="MRI"
                  ):
    def get_input_grad(model, samples): 
        # import pdb 
        # pdb.set_trace()
        outputs = model(samples)
        out_size = outputs.size()
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        print("reading from datapath", args.data_path) 


        test_loader = DataLoader(Test_Data(root_dir=args.data_path, modality_list = [args.modality_name], target_folder="test"), batch_size=1, shuffle=False) 

        model = MODEL
        model.cuda() 
        model.eval()

        optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for idx,data_sample in enumerate(test_loader):
            if meter.count == args.num_images:
                return meter.avg

            samples = data_sample[0] 
            _, _, H, W = samples.size()
            samples = samples.type(torch.FloatTensor).cuda(non_blocking=True)
            samples.requires_grad = True
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples)
            torch.cuda.empty_cache()
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'accumulat{idx}')
                meter.update(contribution_scores)

        return meter.avg


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.data_path = data_path
    # args.save_path = save_path 
    args.modality_name = modality_name
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return main(args, MODEL)


# import pdb 
# pdb.set_trace()

method = "Restore_RWKV"
save_dir = "experiment/{}".format(method) 
data_root = "/home/data/zhiwen/dataset/All-in-One/" 
modality_name = "MRI"


Generator = Restore_RWKV() 
Generator.load_state_dict(torch.load(os.path.join(save_dir, "Model","Generator_best.pth"))) 

grad_map = visualize_erf(Generator, modality_name = modality_name) 
io.save(grad_map, os.path.join(save_dir, "ERF","{}_ERF.bin".format(method)))
analyze_erf(source=grad_map, dest=os.path.join(save_dir, "ERF","{}_ERF.png".format(method)))

        







