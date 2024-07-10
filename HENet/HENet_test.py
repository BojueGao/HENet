# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.mobilevit_loss32_ficture import pvt
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./RGBD_for_test_all/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = pvt()
model.load_state_dict(torch.load('./cpts/mobile_loss_28_3/mobile_vit_epoch_best.pth'))
model.cuda()
model.eval()

# ##################################################  HENet  ############################################

# test_datasets = ['SIP', 'SSD', 'NJU2K', 'NLPR', 'STERE', 'DES', 'LFSD', 'DUT-RGBD']
# test_datasets = ['VT821', 'VT1000', 'VT5000']
test_datasets = ['NJU2K_mmmmm']
# test_datasets = ['NJU2K']
for dataset in test_datasets:
    save_path = './test_maps/middle/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        out1, out2, out3, out4 = model(image, depth)

        out4 = F.upsample(out4, size=gt.shape, mode='bilinear', align_corners=False)

        out = out4.sigmoid().data.cpu().numpy().squeeze()

        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        print('save img to: ', save_path + name)

        cv2.imwrite(save_path + name, out * 255)
    print('Test Done!')

