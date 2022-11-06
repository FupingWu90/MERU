# -*- coding:utf-8 -*-


import math

import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict



from module_modifier import ModuleHelper





model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}





def conv3x3(in_planes, out_planes, stride=1):

    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=1, bias=False)



class Channel_Attention(nn.Module):

    def __init__(self,ratio=16,in_channel=64 ):

        super(Channel_Attention, self).__init__()
        self.ratio = ratio
        self.activate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_channel,in_channel//ratio,kernel_size = 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel // ratio,in_channel,kernel_size = 1),
                                      nn.Sigmoid())


    def forward(self,x):
        actition = self.activate(x)
        out = torch.mul(x,actition)

        return out


class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1,kernel_size = 1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out


def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_channel

        self.f = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channel//8, out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * (C//8) * (W * H)
        self_attetion = self_attetion.view(m_batchsize, -1, width, height)  # B * (C//8) * W * H

        self_attetion = self.v(self_attetion)   # B * C * W * H

        out = self.gamma * self_attetion   #############  +x

        return out


class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, attention_mapping,inplanes, planes, stride=1, downsample=None, norm_type=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)

        self.downsample = downsample

        self.stride = stride

        self.attention = attention_mapping



    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        if self.attention is not None:
            out = self.attention(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, attention_mapping,inplanes, planes, stride=1, downsample=None, norm_type=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,

                               padding=1, bias=False)

        self.bn2 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

        self.attention = attention_mapping



    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)

        if self.attention is not None:
            out = self.attention(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block,attention, layers, num_classes=1000, deep_base=False, norm_type=None,c_scale=1):

        super(ResNet, self).__init__()

        self.inplanes = 128//c_scale if deep_base else 64//c_scale

        if deep_base:

            self.prefix = nn.Sequential(OrderedDict([

                ('conv1', nn.Conv2d(1, 64//c_scale, kernel_size=3, stride=2, padding=1, bias=False)),

                ('bn1', ModuleHelper.BatchNorm2d(norm_type=norm_type)(64//c_scale)),

                ('relu1', nn.ReLU(inplace=False)),

                ('conv2', nn.Conv2d(64//c_scale, 64//c_scale, kernel_size=3, stride=1, padding=1, bias=False)),

                ('bn2', ModuleHelper.BatchNorm2d(norm_type=norm_type)(64//c_scale)),

                ('relu2', nn.ReLU(inplace=False)),

                ('conv3', nn.Conv2d(64//c_scale, 128//c_scale, kernel_size=3, stride=1, padding=1, bias=False)),

                ('bn3', ModuleHelper.BatchNorm2d(norm_type=norm_type)(self.inplanes)),

                ('relu3', nn.ReLU(inplace=False))]

            ))

        else:

            self.prefix = nn.Sequential(OrderedDict([

                ('conv1', nn.Conv2d(1, 64//c_scale, kernel_size=7, stride=2, padding=3, bias=False)),

                ('bn1', ModuleHelper.BatchNorm2d(norm_type=norm_type)(self.inplanes)),

                ('relu', nn.ReLU(inplace=False))]

            ))



        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # change.



        self.layer1 = self._make_layer(block,attention, 64//c_scale, layers[0], norm_type=norm_type)

        self.layer2 = self._make_layer(block,attention, 128//c_scale, layers[1], stride=2, norm_type=norm_type)

        self.layer3 = self._make_layer(block,attention, 256//c_scale, layers[2], stride=2, norm_type=norm_type)

        self.layer4 = self._make_layer(block,attention, 512//c_scale, layers[3], stride=2, norm_type=norm_type)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512//c_scale * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, ModuleHelper.BatchNorm2d(norm_type=norm_type, ret_cls=True)):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def _make_layer(self, block,attention, planes, blocks, stride=1, norm_type=None):

        downsample = None
        attention_mapping = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=1, stride=stride, bias=False),

                ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes * block.expansion),

            )

        if attention=='no_attention':
            attention_mapping = None
        elif attention == 'channel_attention':
            attention_mapping = Channel_Attention(in_channel=planes * block.expansion)
        elif attention == 'spatial_attention':
            attention_mapping = Spatial_Attention(planes * block.expansion)
        elif attention == 'self_attention':
            attention_mapping = Self_Attention(planes * block.expansion)





        layers = []

        layers.append(block(attention_mapping,self.inplanes, planes, stride, downsample, norm_type=norm_type))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(attention_mapping,self.inplanes, planes, norm_type=norm_type))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.prefix(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)



        return x





def resnet18(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

        norm_type (str): choose norm type

    """

    model = ResNet(BasicBlock,attention, [2, 2, 2, 2], num_classes=num_classes, deep_base=False, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def deepbase_resnet18(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(BasicBlock,attention, [2, 2, 2, 2], num_classes=num_classes, deep_base=True, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def resnet34(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-34 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(BasicBlock,attention, [3, 4, 6, 3], num_classes=num_classes, deep_base=False, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def deepbase_resnet34(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-34 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(BasicBlock,attention, [3, 4, 6, 3], num_classes=num_classes, deep_base=True, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def resnet50(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck,attention, [3, 4, 6, 3], num_classes=num_classes, deep_base=False, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def deepbase_resnet50(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm',c_scale=1, **kwargs):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck,attention,[3, 4, 6, 3], num_classes=num_classes, deep_base=True, norm_type=norm_type,c_scale=c_scale)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def resnet101(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck, attention,[3, 4, 23, 3], num_classes=num_classes, deep_base=False, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def deepbase_resnet101(num_classes=1000, attention='no_attention',pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck, attention,[3, 4, 23, 3], num_classes=num_classes, deep_base=True, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def resnet152(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-152 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck, attention,[3, 8, 36, 3], num_classes=num_classes, deep_base=False, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model



def deepbase_resnet152(num_classes=1000,attention='no_attention', pretrained=None, norm_type='batchnorm', **kwargs):

    """Constructs a ResNet-152 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on Places

    """

    model = ResNet(Bottleneck, attention,[3, 8, 36, 3], num_classes=num_classes, deep_base=True, norm_type=norm_type)

    model = ModuleHelper.load_model(model, pretrained=pretrained)

    return model