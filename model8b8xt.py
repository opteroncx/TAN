# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
class MeanShiftY(nn.Conv2d):
    def __init__(self, y_mean, y_std, sign=-1):
        super(MeanShiftY, self).__init__(1, 1, kernel_size=1)
        std = torch.Tensor(y_std)
        self.weight.data = torch.eye(1).view(1, 1, 1, 1)
        self.weight.data.div_(std.view(1, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(y_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class upsample_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample_block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

class make_mix(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_mix, self).__init__()
    # self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=2, bias=False)
    self.relu1 = nn.PReLU()
    self.relu2 = nn.PReLU()
    self.frm = FRM(growthRate,growthRate)
    self.channels = nChannels
    # kernel attention
    # self.ka = KAM(growthRate,growthRate)
    self.krelu =  nn.PReLU()
    self.squeeze = nn.AdaptiveAvgPool2d(1)        
    self.conv_down = nn.Conv2d(growthRate, growthRate // 4, kernel_size=1, bias=False)
    self.conv_up = nn.Conv2d(growthRate // 4, growthRate, kernel_size=1, bias=False)
    self.sig = nn.Sigmoid()
  def forward(self, x):
    # original channel=64,growth rate=32
    y1 = self.relu1(self.conv1(x))
    y2 = self.relu2(self.conv2(x))
    y = y1+y2
    ks = self.squeeze(y)
    cd = self.krelu(self.conv_down(ks))
    cu = self.conv_up(cd)
    w = self.sig(cu)
    y1 = y1*w
    y2 = y2*w
    y = y1+y2
    y = self.frm(y)
    x_left = x[:,0:self.channels-32,:,:]
    x_right = x[:,self.channels-32:self.channels,:,:]
    y_left = y[:,0:32,:,:]
    y_right = y[:,32:64,:,:]
    iden = x_left
    addi = x_right+y_left
    tmp = torch.cat((iden,addi),1)
    out = torch.cat((tmp,y_right),1)
    return out

# Mixed Link Block architecture
class MLB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(MLB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_mix(nChannels_, growthRate))
        nChannels_ += 32
    self.dense_layers = nn.Sequential(*modules)    
    # reshape the channel size
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    self.frm = FRM(nChannels,nChannels)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + self.frm(x)
    return out

class FRM(nn.Module):
    '''The feature recalibration module'''
    def __init__(self,inChannels,outChannels):
        super(FRM, self).__init__()

        self.swish = nn.Sigmoid()
        self.channel_squeeze = nn.AdaptiveAvgPool2d(1)        
        self.conv_down = nn.Conv2d(inChannels * 4, inChannels // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(inChannels // 4, inChannels * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=inChannels * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels * 4, out_channels=outChannels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )
    
    def forward(self, x):
        ex = self.trans1(x)
        out1 = self.channel_squeeze(ex)
        out1 = self.conv_down(out1)
        # swish
        out1 = out1*self.swish(out1)
        out1 = self.conv_up(out1)
        weight = self.sig(out1)
        out=ex*weight
        out=self.trans2(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sub_mean = MeanShiftY([0.43806],[1.0])
        self.add_mean = MeanShiftY([0.43806],[1.0],1)
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        self.u1 = upsample_block(64,64*4)
        self.u2 = upsample_block(64,64*4)
        self.u3 = upsample_block(64,64*4)        
        self.ures1 = upsample_block(64,64*4)
        self.ures2 = upsample_block(64,64*4)
        self.ures3 = upsample_block(64,64*4)        
        self.sa = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_G = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        # add multi supervise
        self.convt_F11 = MLB(64, 4, 64)
        self.convt_F12 = MLB(64, 4, 64)
        self.convt_F13 = MLB(64, 4, 64)
        self.convt_F14 = MLB(64, 4, 64)
        self.convt_F15 = MLB(64, 4, 64)
        self.convt_F16 = MLB(64, 4, 64)
        self.convt_F17 = MLB(64, 4, 64)
        self.convt_F18 = MLB(64, 4, 64)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        def pass1(*args):
            x = args[0]
            x = self.sub_mean(x)
            out = self.relu(self.conv_input(x))
            conv1 = self.conv_input2(out)

            convt_F11 = self.convt_F11(conv1)
            convt_F12 = self.convt_F12(convt_F11)
            convt_F13 = self.convt_F13(convt_F12)
            convt_F14 = self.convt_F14(convt_F13)
            convt_F15 = self.convt_F15(convt_F14)

            convt_F16 = self.convt_F16(convt_F15)
            convt_F17 = self.convt_F17(convt_F16)
            convt_F18 = self.convt_F18(convt_F17)
            return convt_F16,convt_F17,convt_F18,out
        convt_F16,convt_F17,convt_F18,out = checkpoint(pass1, x)

        # x = self.sub_mean(x)
        # out = self.relu(self.conv_input(x))
        # conv1 = self.conv_input2(out)

        # convt_F11 = self.convt_F11(conv1)
        # convt_F12 = self.convt_F12(convt_F11)
        # convt_F13 = self.convt_F13(convt_F12)
        # convt_F14 = self.convt_F14(convt_F13)
        # convt_F15 = self.convt_F15(convt_F14)

        # convt_F16 = self.convt_F16(convt_F15)
        # convt_F17 = self.convt_F17(convt_F16)
        # convt_F18 = self.convt_F18(convt_F17)
       
        # multi supervise
        convt_F = [convt_F16,convt_F17,convt_F18]

        u1 = self.u1(out)
        u2 = self.u2(u1)
        u3 = self.u3(u2)
        u3 = self.convt_shape1(u3)

        HR = []
        HUR = []

        for i in range(len(convt_F)):
            # edge attention
            res1 = self.ures1(convt_F[i])
            res2 = self.ures2(res1)
            res3 = self.ures3(res2)
            G = self.conv_G(res3)
            g1 = G[:,0:64,:,:]
            g2 = G[:,64:128,:,:]
            gu1 = self.sa(G)
            ga = g1*gu1
            # combine
            gc = (ga+g2)/2
            convt_R1 = self.convt_R1(gc)
            tmp = u3 + convt_R1
            tmp = self.add_mean(tmp)
            HR.append(tmp)
            HUR.append(convt_R1)

        return HR,u2,HUR

class ScaleLayer(nn.Module):

   def __init__(self, init_value=0.25):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss
