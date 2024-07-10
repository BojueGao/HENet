# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import sys

from torch.utils.checkpoint import checkpoint

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.mobilevit_loss_33_COME import pvt
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt

# 判断GPU是否可用
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# 加载训练集路径
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
# 加载验证集路径
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path
# 保存log日志
if not os.path.exists(save_path):
    os.makedirs(save_path)
logging.basicConfig(filename=save_path + 'mobile_vit.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("mobile_vit_pairs")
# 创建模型
model = pvt()

# 加载backbone的与训练参数
num_parms = 0
if opt.load_rgb is not None:
    model.load_pre(opt.load_rgb, opt.load_depth)

    print('load rgb_model from ', opt.load_rgb)
    print('load depth_model from ', opt.load_depth)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
# 优化器
optimizer = torch.optim.Adam(params, opt.lr)
# 创建文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 加载训练集数据
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# 加载验证集数据
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
# 求训练集数据长度
total_step = len(train_loader)
# 写日志
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load_rgb:{};load_depth:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_rgb, opt.load_depth,
        save_path,
        opt.decay_epoch))
# 定义损失函数计算函数
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# 定义IOU损失
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


# 开始训练
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    for i, (images, gts, depth) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = images.cuda()
        gts = gts.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        # 放入网络进行训练
        out1, out2, out3, out4 = model(images, depth)

        # 计算损失
        loss1 = F.binary_cross_entropy_with_logits(out1, gts) + iou_loss(out1, gts)
        loss2 = F.binary_cross_entropy_with_logits(out2, gts) + iou_loss(out2, gts)
        loss3 = F.binary_cross_entropy_with_logits(out3, gts) + iou_loss(out3, gts)
        loss4 = F.binary_cross_entropy_with_logits(out4, gts) + iou_loss(out4, gts)
        loss = loss1 + loss2 + loss3 + loss4
        # 反向传播
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step += 1
        epoch_step += 1
        # 最终损失和
        loss_all += loss.data
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss1:{:4f}, loss2:{:4f}, loss3:{:4f}, '
                  'loss4:{:4f}, loss:{:4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         optimizer.state_dict()['param_groups'][0]['lr'], loss1.data, loss2.data, loss3.data, loss4.data, loss.data))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss1:{:4f}, loss2:{:4f}, '
                'loss3:{:4f}, loss4:{:4f}, loss:{:4f} mem_use:{:.0f}MB'.
                format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                       loss1.data, loss2.data, loss3.data, loss4.data, loss.data, memory_used))
            writer.add_scalar('Loss', loss.data, global_step=step)
            grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('Ground_truth', grid_image, step)

            res1 = out1[0].clone()
            res1 = res1.sigmoid().data.cpu().numpy().squeeze()
            res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
            writer.add_image('res1', torch.tensor(res1), step, dataformats='HW')

            res2 = out2[0].clone()
            res2 = res2.sigmoid().data.cpu().numpy().squeeze()
            res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
            writer.add_image('res2', torch.tensor(res2), step, dataformats='HW')

            res3 = out3[0].clone()
            res3 = res3.sigmoid().data.cpu().numpy().squeeze()
            res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)
            writer.add_image('res3', torch.tensor(res3), step, dataformats='HW')

            res4 = out4[0].clone()
            res4 = res4.sigmoid().data.cpu().numpy().squeeze()
            res4 = (res4 - res4.min()) / (res4.max() - res4.min() + 1e-8)
            writer.add_image('res4', torch.tensor(res4), step, dataformats='HW')

    loss_all /= epoch_step
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    # 每5轮保存一次模型参数
    if (epoch) % 20 == 0:
        torch.save(model.state_dict(), save_path + 'mobile_vit_epoch_{}.pth'.format(epoch))


# 验证
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_x1 = 0
        mae_x2 = 0
        mae_x3 = 0
        mae_x4 = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            x1, x2, x3, x4 = model(image, depth)

            res1 = F.upsample(x1, size=gt.shape, mode='bilinear', align_corners=False)
            res1 = res1.sigmoid().data.cpu().numpy().squeeze()
            res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
            # 计算MAE
            mae_x1 += np.sum(np.abs(res1 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            res2 = F.upsample(x2, size=gt.shape, mode='bilinear', align_corners=False)
            res2 = res2.sigmoid().data.cpu().numpy().squeeze()
            res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
            # 计算MAE
            mae_x2 += np.sum(np.abs(res2 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            res3 = F.upsample(x3, size=gt.shape, mode='bilinear', align_corners=False)
            res3 = res3.sigmoid().data.cpu().numpy().squeeze()
            res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)
            # 计算MAE
            mae_x3 += np.sum(np.abs(res3 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            res4 = F.upsample(x4, size=gt.shape, mode='bilinear', align_corners=False)
            res4 = res4.sigmoid().data.cpu().numpy().squeeze()
            res4 = (res4 - res4.min()) / (res4.max() - res4.min() + 1e-8)
            # 计算MAE
            mae_x4 += np.sum(np.abs(res4 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae_sum = (mae_x1 + mae_x2 + mae_x3 + mae_x4) / 4
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        # 进行MAE比较
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'mobile_vit_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


# def bce2d_new(input, target, reduction=None):
#     assert (input.size() == target.size())
#     pos = torch.eq(target, 1).float()
#     neg = torch.eq(target, 0).float()
#
#     num_pos = torch.sum(pos)
#     num_neg = torch.sum(neg)
#     num_total = num_pos + num_neg
#
#     alpha = num_neg / num_total
#     beta = 1.1 * num_pos / num_total
#     weights = alpha * pos + beta * neg
#
#     return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
