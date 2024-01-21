import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.z221003_B_add_FI import Network
#from net import DFMNet
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=int, default=1, help='select gpu id')
parser.add_argument('--test_path',type=str,default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0,1')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#load the model
model = Network()
#model = DFMNet()
model.load_state_dict(torch.load('/media/lab509-1/data/TYY/RGBD-COD/ISM22.04.13/train_pth/1080Ti/z221003_B_add_FI/Net_epoch_best.pth'),False)
#model = torch.nn.DataParallel(model)    
#model.to(device)
model.cuda()
model.eval()

#test
test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']
#test_datasets = ['NC4K']
for dataset in test_datasets:
    save_path = './test_maps/1080Ti/z221003_B_add_FI/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth2/'
    #edge_root=dataset_path +dataset +'/edge/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        res= model(image, depth)
        # 单监督
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # 多监督
        #res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
