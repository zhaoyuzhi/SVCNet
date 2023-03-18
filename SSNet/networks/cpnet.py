import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *
import networks.network_utils as network_utils

#-----------------------------------------------
#               Generator - CPNet
#-----------------------------------------------
class CPNet_Seg_subnet(nn.Module):
    def __init__(self, opt):
        super(CPNet_Seg_subnet, self).__init__()
        self.opt = opt
        self.conv1_1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv1_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv2_1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv2_2 = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv3_1 = Conv2dLayer(opt.start_channels * 3, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv3_2 = Conv2dLayer(opt.start_channels, opt.seg_channels, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, input1, input2, input3):

        # dimension of input 1: batch * 256 * 32 * 32
        # dimension of input 2: batch * 128 * 64 * 64
        # dimension of input 3: batch * 64 * 128 * 128

        # 64*64
        input1_64 = F.interpolate(input1, scale_factor = 2, mode = 'bilinear')
        input1_64 = self.conv1_1(input1_64)                     # out: batch * 128 * 64 * 64
        input2 = torch.cat((input1_64, input2), 1)              # out: batch * 256 * 64 * 64
        input2 = self.conv2_1(input2)                           # out: batch * 64 * 64 * 64

        # 128*128
        input1_128 = F.interpolate(input1, scale_factor = 4, mode = 'bilinear')
        input1_128 = self.conv1_2(input1_128)                   # out: batch * 64 * 128 * 128
        input2 = F.interpolate(input2, scale_factor = 2, mode = 'bilinear')
        input2 = self.conv2_2(input2)                           # out: batch * 64 * 128 * 128
        input3 = torch.cat((input1_128, input2, input3), 1)     # out: batch * 192 * 128 * 128
        input3 = self.conv3_1(input3)                           # out: batch * 64 * 128 * 128
        
        # 256*256
        input3 = F.interpolate(input3, scale_factor = 2, mode = 'bilinear')
        out = self.conv3_2(input3)                              # out: batch * 1 * 256 * 256
        return out

class CPNet_VGG16(nn.Module):
    def __init__(self, opt):
        super(CPNet_VGG16, self).__init__()
        self.opt = opt
        # VGG Encoder (*gray for grayscale / *ref for reference RGB)
        self.vgg = network_utils.create_vggnet(opt.vgg_name)
        self.conv1_2_gray = []
        self.conv2_2_gray = []
        self.conv3_3_gray = []
        self.conv4_3_gray = []
        self.conv5_3_gray = []
        for i in range(0, 4):
            self.conv1_2_gray.append(self.vgg.features[i])
        for i in range(4, 9):
            self.conv2_2_gray.append(self.vgg.features[i])
        for i in range(9, 16):
            self.conv3_3_gray.append(self.vgg.features[i])
        for i in range(16, 23):
            self.conv4_3_gray.append(self.vgg.features[i])
        for i in range(23, 29):
            self.conv5_3_gray.append(self.vgg.features[i])
        self.conv1_2_gray = nn.Sequential(*self.conv1_2_gray)
        self.conv2_2_gray = nn.Sequential(*self.conv2_2_gray)
        self.conv3_3_gray = nn.Sequential(*self.conv3_3_gray)
        self.conv4_3_gray = nn.Sequential(*self.conv4_3_gray)
        self.conv5_3_gray = nn.Sequential(*self.conv5_3_gray)
        # Encoder
        self.E1 = nn.Sequential(
            Conv2dLayer(opt.in_channels + opt.scribble_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E2 = nn.Sequential(
            Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E3 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E4 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E5 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        # latent feature fusion (*gray for grayscale / *ref for reference RGB)
        self.shortcut1 = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut4 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut5 = Conv2dLayer(opt.start_channels * 8 + 512, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T7 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T8 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Decoder
        self.D5_1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D4_1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D3_1 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D2_1 = TransposeConv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D5_2 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D4_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D3_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D2_2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D1 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, x, scribble):
        # grayscale vgg encoder (grayscale features)
        gray_fea = torch.cat((x, x, x), 1)                      # out: batch * 3 * 256 * 256
        gray_fea1 = self.conv1_2_gray(gray_fea)                 # out: batch * 64 * 256 * 256
        gray_fea2 = self.conv2_2_gray(gray_fea1)                # out: batch * 128 * 128 * 128
        gray_fea3 = self.conv3_3_gray(gray_fea2)                # out: batch * 256 * 64 * 64
        gray_fea4 = self.conv4_3_gray(gray_fea3)                # out: batch * 512 * 32 * 32
        gray_fea5 = self.conv5_3_gray(gray_fea4)                # out: batch * 512 * 16 * 16
        # grayscale encoder (grayscale + scribble features)
        comb_fea = torch.cat((x, scribble), 1)                  # out: batch * 3 * 256 * 256
        x1 = self.E1(comb_fea)                                  # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16
        # bottleneck
        dec5 = torch.cat((x5, gray_fea5), 1)
        dec5 = self.shortcut5(dec5)                             # out: batch * 512 * 16 * 16
        dec5 = self.T1(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T2(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T3(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T4(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T5(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T6(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T7(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T8(dec5)                                    # out: batch * 512 * 16 * 16
        # decoder
        # decoder level 5
        shortcut4 = self.shortcut4(x4)                          # out: batch * 512 * 32 * 32
        dec4 = self.D5_1(dec5)                                  # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec4, shortcut4), 1)                  # out: batch * 1024 * 32 * 32
        dec4 = self.D5_2(dec4)                                  # out: batch * 256 * 32 * 32
        # decoder level 4
        shortcut3 = self.shortcut3(x3)                          # out: batch * 256 * 64 * 64
        dec3 = self.D4_1(dec4)                                  # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec3, shortcut3), 1)                  # out: batch * 512 * 64 * 64
        dec3 = self.D4_2(dec3)                                  # out: batch * 128 * 64 * 64
        # decoder level 3
        shortcut2 = self.shortcut2(x2)                          # out: batch * 128 * 128 * 128
        dec3 = self.D3_1(dec3)                                  # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, shortcut2), 1)                  # out: batch * 256 * 128 * 128
        dec2 = self.D3_2(dec2)                                  # out: batch * 64 * 128 * 128
        # decoder level 2
        shortcut1 = self.shortcut1(x1)                          # out: batch * 64 * 256 * 256
        dec1 = self.D2_1(dec2)                                  # out: batch * 64 * 256 * 256
        dec1 = torch.cat((dec1, shortcut1), 1)                  # out: batch * 128 * 256 * 256
        dec1 = self.D2_2(dec1)                                  # out: batch * 64 * 256 * 256
        # decoder level 1
        out = self.D1(dec1)                                     # out: batch * 2 * 256 * 256
        return out, out

class CPNet_VGG16_Seg(nn.Module):
    def __init__(self, opt):
        super(CPNet_VGG16_Seg, self).__init__()
        self.opt = opt
        # VGG Encoder (*gray for grayscale / *ref for reference RGB)
        self.vgg = network_utils.create_vggnet(opt.vgg_name)
        self.conv1_2_gray = []
        self.conv2_2_gray = []
        self.conv3_3_gray = []
        self.conv4_3_gray = []
        self.conv5_3_gray = []
        for i in range(0, 4):
            self.conv1_2_gray.append(self.vgg.features[i])
        for i in range(4, 9):
            self.conv2_2_gray.append(self.vgg.features[i])
        for i in range(9, 16):
            self.conv3_3_gray.append(self.vgg.features[i])
        for i in range(16, 23):
            self.conv4_3_gray.append(self.vgg.features[i])
        for i in range(23, 29):
            self.conv5_3_gray.append(self.vgg.features[i])
        self.conv1_2_gray = nn.Sequential(*self.conv1_2_gray)
        self.conv2_2_gray = nn.Sequential(*self.conv2_2_gray)
        self.conv3_3_gray = nn.Sequential(*self.conv3_3_gray)
        self.conv4_3_gray = nn.Sequential(*self.conv4_3_gray)
        self.conv5_3_gray = nn.Sequential(*self.conv5_3_gray)
        # Encoder
        self.E1 = nn.Sequential(
            Conv2dLayer(opt.in_channels + opt.scribble_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E2 = nn.Sequential(
            Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E3 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E4 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        self.E5 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        )
        # latent feature fusion (*gray for grayscale / *ref for reference RGB)
        self.shortcut1 = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut4 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.shortcut5 = Conv2dLayer(opt.start_channels * 8 + 512, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T7 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.T8 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Decoder
        self.D5_1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D4_1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D3_1 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D2_1 = TransposeConv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g, scale_factor = 2)
        self.D5_2 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D4_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D3_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D2_2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.D1 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        # Decoder - segmentation
        self.seg_decoder = CPNet_Seg_subnet(opt)

    def forward(self, x, scribble):
        # grayscale vgg encoder (grayscale features)
        gray_fea = torch.cat((x, x, x), 1)                      # out: batch * 3 * 256 * 256
        gray_fea1 = self.conv1_2_gray(gray_fea)                 # out: batch * 64 * 256 * 256
        gray_fea2 = self.conv2_2_gray(gray_fea1)                # out: batch * 128 * 128 * 128
        gray_fea3 = self.conv3_3_gray(gray_fea2)                # out: batch * 256 * 64 * 64
        gray_fea4 = self.conv4_3_gray(gray_fea3)                # out: batch * 512 * 32 * 32
        gray_fea5 = self.conv5_3_gray(gray_fea4)                # out: batch * 512 * 16 * 16
        # grayscale encoder (grayscale + scribble features)
        comb_fea = torch.cat((x, scribble), 1)                  # out: batch * 3 * 256 * 256
        x1 = self.E1(comb_fea)                                  # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16
        # bottleneck
        dec5 = torch.cat((x5, gray_fea5), 1)
        dec5 = self.shortcut5(dec5)                             # out: batch * 512 * 16 * 16
        dec5 = self.T1(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T2(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T3(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T4(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T5(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T6(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T7(dec5)                                    # out: batch * 512 * 16 * 16
        dec5 = self.T8(dec5)                                    # out: batch * 512 * 16 * 16
        # decoder
        # decoder level 5
        shortcut4 = self.shortcut4(x4)                          # out: batch * 512 * 32 * 32
        dec4 = self.D5_1(dec5)                                  # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec4, shortcut4), 1)                  # out: batch * 1024 * 32 * 32
        dec4 = self.D5_2(dec4)                                  # out: batch * 256 * 32 * 32
        dec4_copy = dec4
        # decoder level 4
        shortcut3 = self.shortcut3(x3)                          # out: batch * 256 * 64 * 64
        dec3 = self.D4_1(dec4)                                  # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec3, shortcut3), 1)                  # out: batch * 512 * 64 * 64
        dec3 = self.D4_2(dec3)                                  # out: batch * 128 * 64 * 64
        dec3_copy = dec3
        # decoder level 3
        shortcut2 = self.shortcut2(x2)                          # out: batch * 128 * 128 * 128
        dec3 = self.D3_1(dec3)                                  # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, shortcut2), 1)                  # out: batch * 256 * 128 * 128
        dec2 = self.D3_2(dec2)                                  # out: batch * 64 * 128 * 128
        dec2_copy = dec2
        # decoder level 2
        shortcut1 = self.shortcut1(x1)                          # out: batch * 64 * 256 * 256
        dec1 = self.D2_1(dec2)                                  # out: batch * 64 * 256 * 256
        dec1 = torch.cat((dec1, shortcut1), 1)                  # out: batch * 128 * 256 * 256
        dec1 = self.D2_2(dec1)                                  # out: batch * 64 * 256 * 256
        # decoder level 1
        out = self.D1(dec1)                                     # out: batch * 2 * 256 * 256
        # seg decoder
        seg_out = self.seg_decoder(dec4_copy, dec3_copy, dec2_copy)
        return out, seg_out

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--start2_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'visible mask channel')
    parser.add_argument('--warn_channels', type = int, default = 16, help = 'start channel for Warp_Artifact_Removal_Net')
    parser.add_argument('--transformer_channels', type = int, default = 32, help = 'start channel for Transformer')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_warn', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_warn', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    parser.add_argument('--vgg_name', type = str, \
        default = "F:\\submitted papers\\retrieval_meets_colorization\\code\\util\\vgg16_pretrained.pth", \
            help = 'load the pre-trained vgg model with certain epoch')
    opt = parser.parse_args()

    net = CPNet_VGG16_Seg(opt).cuda()
    a = torch.randn(1, 1, 128, 256).cuda()
    s = torch.randn(1, 2, 128, 256).cuda()
    b, seg = net(a, s)
    print(b.shape, seg.shape)
