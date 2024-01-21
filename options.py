import argparse
#1.创建解析器
# 使用 argparse 的第一步是创建一个 ArgumentParser 对象。
# ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
parser = argparse.ArgumentParser()
#添加参数
parser.add_argument('--epoch', type=int, default=201, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
#parser.add_argument('--load', type=str, default='./model_pths/BBSNet.pth', help='train from checkpoints')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/TrainDataset/image/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/TrainDataset/depth2/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/TrainDataset/GT/', help='the training gt images root')
#parser.add_argument('--edge_root', type=str, default='../BBS_dataset/RGBD_for_train/edge/', help='the training edge images root')
parser.add_argument('--test_rgb_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/test/CAMO/image/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/test/CAMO/depth2/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default='/media/lab509-1/data/TYY/RGBD-COD/Dataset/test/CAMO/GT/', help='the test gt images root')
#parser.add_argument('--test_edge_root', type=str, default='../BBS_dataset/test_in_train/edge/', help='the test edge images root')
parser.add_argument('--save_path', type=str, default='./train_pth/1080Ti/z221003_B_add_FI/', help='the path to save models and logs')
#解析参数
opt = parser.parse_args()
