# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/5/16 14:52

"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load_rgb', type=str, default='./mobilevit_s.pt', help='train from checkpoints')
parser.add_argument('--load_depth', type=str, default='./mobilevit_s.pt', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./RGBD_train/COME/train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='./RGBD_train/COME/train/Depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./RGBD_train/COME/train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='./RGBD_train/COME/test/COME-E/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='./RGBD_train/COME/test/COME-E/depths/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./RGBD_train/COME/test/COME-E/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./cpts/mobile_loss_COME_E/', help='the path to save models and logs')
opt = parser.parse_args()
