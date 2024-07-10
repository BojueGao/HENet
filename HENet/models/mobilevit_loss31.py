# -*- coding: utf-8 -*-

import os
from functools import partial

import cv2
import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math
from mobilevit import mobile_vit_xx_small
from torchvision.ops import DeformConv2d


def upsample(x, y):
    return F.interpolate(x, y, mode='bilinear', align_corners=True)


class BConv(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, h, w, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, w))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(kernel_size=self.sz, stride=self.sz)
        self.gmpLayer = nn.MaxPool2d(kernel_size=self.sz, stride=self.sz)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg


class sa_layer(nn.Module):
    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out


class Fusion(nn.Module):
    def __init__(self, channel, h, w):
        super(Fusion, self).__init__()
        self.convTo2 = nn.Conv2d(channel * 2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.h = h
        self.w = w
        self.coordAttention = CoordAtt(channel, channel, self.h, self.w)
        self.channel = channel

        self.glbamp = AP_MP()
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 2 + 1, channel, 1),
            nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, r, d):
        H = torch.cat((r, d), dim=1)
        H_conv = self.sig(self.convTo2(H))
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        Ga = r * ga
        Gm = d * gm

        Gm_out = self.coordAttention(Gm)
        res = Gm_out + Ga

        gamp = self.upsample2(self.glbamp(res))
        gamp = gamp / math.sqrt(self.channel)

        cat = torch.cat((Ga, Gm_out, gamp), dim=1)
        cat = self.conv_cat(cat)
        sal = res + cat

        return sal


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decoder, self).__init__()
        self.DWConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.BConv = BConv(out_channel * 2, out_channel, 3, 1, 1)

    def forward(self, x):
        x_left = self.conv(x)
        x_right = self.DWConv(x)
        out = self.BConv(torch.cat((x_left, x_right), dim=1))

        return out

class pvt(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(pvt, self).__init__()

        # encoder
        self.rgb_pvt = mobile_vit_xx_small()
        self.depth_pvt = mobile_vit_xx_small()

        self.fusion1 = Fusion(80, 12, 12)
        self.fusion2 = Fusion(64, 24, 24)
        self.fusion3 = Fusion(48, 48, 48)
        self.fusion4 = Fusion(24, 96, 96)

        # decoder
        self.conv_x4 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x2 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x1 = nn.Conv2d(32, 1, 3, 1, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder_12 = decoder(80, 32)
        self.sh_attention_12 = sa_layer(32)

        self.decoder_24 = decoder(64, 32)
        self.sh_attention_24 = sa_layer(32)
        self.bconv24 = BConv(64, 32, 3, 1, 1)

        self.decoder_48 = decoder(48, 32)
        self.sh_attention_48 = sa_layer(32)
        self.bconv48 = BConv(64, 32, 3, 1, 1)

        self.decoder_96 = decoder(24, 32)
        self.sh_attention_96 = sa_layer(32)
        self.bconv96 = BConv(64, 32, 3, 1, 1)

    def forward(self, rgb, depth):
        rgb_list = self.rgb_pvt(rgb)
        depth_list = self.depth_pvt(depth)

        r4 = rgb_list[1]  # (8,64,96,96)
        r3 = rgb_list[2]  # (8,96,48,48)
        r2 = rgb_list[3]  # (8,128,24,24)
        r1 = rgb_list[4]  # (8,160,12,12)

        d4 = depth_list[1]  # (8,64,96,96)
        d3 = depth_list[2]  # (8,96,48,48)
        d2 = depth_list[3]  # (8,128,24,24)
        d1 = depth_list[4]  # (8,160,12,12)

        # Encoder
        F1 = self.fusion1(r1, d1)  # (160,12,12)
        F2 = self.fusion2(r2, d2)  # (128,24,24)
        F3 = self.fusion3(r3, d3)  # (96,48,48)
        F4 = self.fusion4(r4, d4)  # (64,96,96)

        # Decoder
        x1_decoder = self.decoder_12(F1)
        x1 = self.sh_attention_12(x1_decoder)

        x2_decoder = self.decoder_24(F2)
        x2_att = self.sh_attention_24(x2_decoder)
        x2_mul = self.upsample2(x1_decoder) * x2_att
        x2_cat = self.bconv24(torch.cat((x2_mul, x2_att), dim=1))
        x2 = x2_cat + self.upsample2(x1)

        x3_decoder = self.decoder_48(F3)
        x3_att = self.sh_attention_48(x3_decoder)
        x3_mul = self.upsample2(x2_decoder) * x3_att
        x3_cat = self.bconv48(torch.cat((x3_mul, x3_att), dim=1))
        x3 = x3_cat + self.upsample2(x2)

        x4_decoder = self.decoder_96(F4)
        x4_att = self.sh_attention_96(x4_decoder)
        x4_mul = self.upsample2(x3_decoder) * x4_att
        x4_cat = self.bconv96(torch.cat((x4_mul, x4_att), dim=1))
        x4 = x4_cat + self.upsample2(x3)

        shape = rgb.size()[2:]  # shape:(384,384)

        out1 = F.interpolate(self.conv_x1(x1), size=shape, mode='bilinear')  # (b,1,384,384)
        out2 = F.interpolate(self.conv_x2(x2), size=shape, mode='bilinear')  # (b,1,384,384)
        out3 = F.interpolate(self.conv_x3(x3), size=shape, mode='bilinear')  # (b,1,384,384)
        out4 = F.interpolate(self.conv_x4(x4), size=shape, mode='bilinear')  # (b,1,384,384)

        return out1, out2, out3, out4

        # ###################################################   end   #########################################

    def load_pre(self, pre_model_rgb, pre_model_depth):
        self.rgb_pvt.load_state_dict(torch.load(pre_model_rgb), strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model_rgb}")
        self.depth_pvt.load_state_dict(torch.load(pre_model_depth), strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model_depth}")

    # def load_pre(self, model):
    #     self.edge_layer.load_state_dict(model,strict=False)

# ########################################### end #####################################################
import torch
from torchvision.models import resnet18
from thop import profile

model = pvt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input1 = torch.randn(1, 3, 384, 384)
input1 = input1.to(device)
input2 = torch.randn(1, 3, 384, 384)
input2 = input2.to(device)
flops, params = profile(model.cuda(), inputs=(input1, input2,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))

import torch


iterations = 300   # 重复计算的轮次

model = pvt()
device = torch.device("cuda:0")
model.to(device)

random_input1 = torch.randn(1, 3, 384, 384).to(device)
random_input2 = torch.randn(1, 3, 384, 384).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input1,random_input2)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input1,random_input2)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
