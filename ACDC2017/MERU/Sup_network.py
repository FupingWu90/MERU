# -*- coding:utf-8 -*-

import math, time

from itertools import chain

import torch

import torch.nn.functional as F

from torch import nn

from decoder import *

from encoder import Encoder


class ResSeg(nn.Module):
    def __init__(self,resnet_type,attention_type,num_out_ch,num_class,upscale,pretrained,c_scale):
        super(ResSeg, self).__init__()

        self.encoder = Encoder(resnet_type,attention_type,pretrained,c_scale)

        decoder_in_ch = num_out_ch//c_scale // 4   ### PSP mdule

        self.decoder = MainDecoder(upscale,decoder_in_ch,num_class)


    def forward(self,x):
        x = self.encoder(x)
        out = self.decoder(x)

        return out

class DoubleResSeg(nn.Module):
    def __init__(self,resnet_type1,attention_type1,resnet_type2,attention_type2,num_out_ch,num_class,upscale,pretrained):
        super(DoubleResSeg, self).__init__()

        self.seg_net1 = ResSeg(resnet_type1,attention_type1,num_out_ch,num_class,upscale,pretrained)
        self.seg_net2 = ResSeg(resnet_type2, attention_type2, num_out_ch, num_class, upscale, pretrained)


    def forward(self,x):
        out1 = self.seg_net1(x)
        out2 = self.seg_net2(x)

        return out1,out2



class GroupConvSegNet(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(GroupConvSegNet,self).__init__()

        self.seg_net = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=out_channel),
                                         nn.BatchNorm2d(in_channel),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(in_channel, in_channel//2, kernel_size=3, padding=1, groups=out_channel),
                                         nn.BatchNorm2d(in_channel//2),
                                         nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channel//2, in_channel // 2, kernel_size=3, padding=1,
                                               groups=out_channel),
                                     nn.BatchNorm2d(in_channel // 2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channel // 2, out_channel, kernel_size=3, padding=1,
                                               groups=out_channel),
                                     )

    def forward(self,x):
        x = self.seg_net(x)

        return x



class UNet2d(nn.Module):
    def __init__(self,KERNEL=3,PADDING=1):
        super(UNet2d,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convt1=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.convt2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.convt3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.convt4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)

        self.conv_seq1 = nn.Sequential( nn.Conv2d(1,64,kernel_size=KERNEL,padding=PADDING),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64,64,kernel_size=KERNEL,padding=PADDING),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(1024, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU(inplace=True))


        self.deconv_seq1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))
        self.segdown4_seq = GroupConvSegNet(256,4)

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.segdown2_seq = GroupConvSegNet(128,4)

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       )

        self.seg_seq  = GroupConvSegNet(64,4)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.segfusion = nn.Sequential(nn.Conv2d(4 * 3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING), )

        self.soft = nn.Softmax2d()

        self.sigmiod = nn.Sigmoid()




    def forward(self,x):
        out1 = self.conv_seq1(x)
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5),out4),1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1),out3),1))
        segout_down4 = self.segdown4_seq(deout2)

        deout3= self.deconv_seq3(torch.cat((self.convt3(deout2),out2),1))
        segout_down2 = self.segdown2_seq(deout3)

        deout4= self.deconv_seq4(torch.cat((self.convt4(deout3),out1),1))
        out = self.seg_seq(deout4)

        fusion_seg = self.segfusion(torch.cat((self.sigmiod(out), self.upsample2(self.sigmiod(segout_down2)), self.upsample4(self.sigmiod(segout_down4))), dim=1))

        prediction = self.soft(fusion_seg)

        pi_all = torch.mean(prediction,(2,3))



        return segout_down4,segout_down2,out,fusion_seg,pi_all,out5

class PriorPredNet(nn.Module):
    def __init__(self,):
        super(PriorPredNet,self).__init__()
        self.prior_pred = nn.Sequential(nn.Linear(1024 * 6 * 6, 1024*3),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024*3, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 4),
                                        )

    def forward(self, x):
        pi = self.prior_pred(x.view(x.size(0), -1))

        return pi
