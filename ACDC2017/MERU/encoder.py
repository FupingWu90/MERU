#!/usr/bin/env python

# -*- coding:utf-8 -*-

# Author: Donny You(youansheng@gmail.com)




import torch

import torch.nn as nn

import torch.nn.functional as F

import os


from models import *

resnet50 = {

    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",

}

class NormalResnetBackbone(nn.Module):

    def __init__(self, orig_resnet):

        super(NormalResnetBackbone, self).__init__()



        self.num_features = 2048

        # take pretrained resnet, except AvgPool and FC

        self.prefix = orig_resnet.prefix

        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1

        self.layer2 = orig_resnet.layer2

        self.layer3 = orig_resnet.layer3

        self.layer4 = orig_resnet.layer4



    def get_num_features(self):

        return self.num_features



    def forward(self, x):

        tuple_features = list()

        x = self.prefix(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        tuple_features.append(x)

        x = self.layer2(x)

        tuple_features.append(x)

        x = self.layer3(x)

        tuple_features.append(x)

        x = self.layer4(x)

        tuple_features.append(x)



        return tuple_features





class DilatedResnetBackbone(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):

        super(DilatedResnetBackbone, self).__init__()



        self.num_features = 2048

        from functools import partial



        if dilate_scale == 8:

            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))

            if multi_grid is None:

                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

            else:

                for i, r in enumerate(multi_grid):

                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))



        elif dilate_scale == 16:

            if multi_grid is None:

                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

            else:

                for i, r in enumerate(multi_grid):

                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))



        # Take pretrained resnet, except AvgPool and FC

        self.prefix = orig_resnet.prefix

        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1

        self.layer2 = orig_resnet.layer2

        self.layer3 = orig_resnet.layer3

        self.layer4 = orig_resnet.layer4



    def _nostride_dilate(self, m, dilate):

        classname = m.__class__.__name__

        if classname.find('Conv') != -1:

            # the convolution with stride

            if m.stride == (2, 2):

                m.stride = (1, 1)

                if m.kernel_size == (3, 3):

                    m.dilation = (dilate // 2, dilate // 2)

                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions

            else:

                if m.kernel_size == (3, 3):

                    m.dilation = (dilate, dilate)

                    m.padding = (dilate, dilate)



    def get_num_features(self):

        return self.num_features



    def forward(self, x):

        tuple_features = list()

        x = self.prefix(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        tuple_features.append(x)

        x = self.layer2(x)

        tuple_features.append(x)

        x = self.layer3(x)

        tuple_features.append(x)

        x = self.layer4(x)

        tuple_features.append(x)



        return tuple_features





def ResNetBackbone(backbone=None,attention='no_attention', pretrained=None, multi_grid=None, norm_type='batchnorm',c_scale=1):

    arch =  backbone

    if arch == 'resnet34':

        orig_resnet = resnet34(pretrained=pretrained)

        arch_net = NormalResnetBackbone(orig_resnet)

        arch_net.num_features = 512



    elif arch == 'resnet34_dilated8':

        orig_resnet = resnet34(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        arch_net.num_features = 512



    elif arch == 'resnet34_dilated16':

        orig_resnet = resnet34(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        arch_net.num_features = 512



    elif arch == 'resnet50':

        orig_resnet = resnet50(pretrained=pretrained)

        arch_net = NormalResnetBackbone(orig_resnet)



    elif arch == 'resnet50_dilated8':

        orig_resnet = resnet50(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)



    elif arch == 'resnet50_dilated16':

        orig_resnet = resnet50(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)



    elif arch == 'deepbase_resnet50':

        if pretrained:

            pretrained = 'models/backbones/pretrained/3x3resnet50-imagenet.pth'

        orig_resnet = deepbase_resnet50(attention=attention,pretrained=pretrained,c_scale=c_scale)

        arch_net = NormalResnetBackbone(orig_resnet)



    elif arch == 'deepbase_resnet50_dilated8':

        if pretrained:

            pretrained = 'models/backbones/pretrained/3x3resnet50-imagenet.pth'

        orig_resnet = deepbase_resnet50(attention=attention,pretrained=pretrained,c_scale=c_scale)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)



    elif arch == 'deepbase_resnet50_dilated16':

        orig_resnet = deepbase_resnet50(attention=attention,pretrained=pretrained,c_scale=c_scale)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)



    elif arch == 'resnet101':

        orig_resnet = resnet101(pretrained=pretrained)

        arch_net = NormalResnetBackbone(orig_resnet)



    elif arch == 'resnet101_dilated8':

        orig_resnet = resnet101(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)



    elif arch == 'resnet101_dilated16':

        orig_resnet = resnet101(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)



    elif arch == 'deepbase_resnet101':

        orig_resnet = deepbase_resnet101(pretrained=pretrained)

        arch_net = NormalResnetBackbone(orig_resnet)



    elif arch == 'deepbase_resnet101_dilated8':

        if pretrained:

            pretrained = 'models/backbones/pretrained/3x3resnet101-imagenet.pth'

        orig_resnet = deepbase_resnet101(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)



    elif arch == 'deepbase_resnet101_dilated16':

        orig_resnet = deepbase_resnet101(pretrained=pretrained)

        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)



    else:

        raise Exception('Architecture undefined!')



    return arch_net





class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)

        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])

        self.bottleneck = nn.Sequential(

            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,

                      kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        bn = nn.BatchNorm2d(out_channels)

        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]

        pyramids = [features]

        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',

                                       align_corners=False) for stage in self.stages])

        output = self.bottleneck(torch.cat(pyramids, dim=1))

        return output


class Encoder(nn.Module):
    def __init__(self,resnet_type, attention_type, pretrained,c_scale):
        super(Encoder, self).__init__()

        if pretrained and not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")

            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone=resnet_type,attention=attention_type, pretrained=pretrained,c_scale=c_scale)

        self.base = nn.Sequential(

            nn.Sequential(model.prefix, model.maxpool),

            model.layer1,

            model.layer2,

            model.layer3,

            model.layer4

        )

        self.psp = _PSPModule(2048//c_scale, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)

        x = self.psp(x)

        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()