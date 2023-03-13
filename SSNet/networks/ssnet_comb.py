import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *
from networks.cpnet import *

# ----------------------------------------
#          Combination network
# ----------------------------------------
class Combination_Net(nn.Module):
    def __init__(self, opt):
        super(Combination_Net, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.out_channels * 9, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E2 = Conv2dLayer(opt.start_channels_comb, opt.start_channels_comb * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E3 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E4 = Conv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E5 = Conv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T2 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T3 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T4 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Decoder
        self.D5 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels_comb * 16, opt.start_channels_comb * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D1 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.outconv = Conv2dLayer(opt.start_channels_comb, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')
        
    def forward(self, p_t, warped_0, warn_outs):
        
        # Input:
        concat_input = torch.cat((p_t, warped_0, warn_outs), 1)

        # encoder
        x1 = self.E1(concat_input)                              # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16

        # bottleneck
        x5 = self.T1(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T2(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T3(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T4(x5)                                        # out: batch * 512 * 16 * 16

        # decoder
        # decoder level 5
        dec5 = self.D5(x5)                                      # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec5, x4), 1)                         # out: batch * 1024 * 32 * 32
        # decoder level 4
        dec4 = self.D4(dec4)                                    # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec4, x3), 1)                         # out: batch * 512 * 64 * 64
        # decoder level 3
        dec3 = self.D3(dec3)                                    # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, x2), 1)                         # out: batch * 256 * 128 * 128
        # decoder level 2
        dec2 = self.D2(dec2)                                    # out: batch * 64 * 128 * 128
        dec1 = torch.cat((dec2, x1), 1)                         # out: batch * 128 * 256 * 256
        dec1 = self.D1(dec1)                                    # out: batch * 64 * 256 * 256
        # decoder level 1
        residual = self.outconv(dec1)                           # out: batch * 2 * 256 * 256
        
        out = warped_0 - residual
        
        return out, residual

# ----------------------------------------
# Combination network (for ablation study)
# ----------------------------------------
class Combination_Net_without_short(nn.Module):
    def __init__(self, opt):
        super(Combination_Net_without_short, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.out_channels * 2, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E2 = Conv2dLayer(opt.start_channels_comb, opt.start_channels_comb * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E3 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E4 = Conv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E5 = Conv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T2 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T3 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T4 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Decoder
        self.D5 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels_comb * 16, opt.start_channels_comb * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D1 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.outconv = Conv2dLayer(opt.start_channels_comb, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')
        
    def forward(self, p_t, warped_0):
        
        # Input:
        concat_input = torch.cat((p_t, warped_0), 1)

        # encoder
        x1 = self.E1(concat_input)                              # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16

        # bottleneck
        x5 = self.T1(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T2(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T3(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T4(x5)                                        # out: batch * 512 * 16 * 16

        # decoder
        # decoder level 5
        dec5 = self.D5(x5)                                      # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec5, x4), 1)                         # out: batch * 1024 * 32 * 32
        # decoder level 4
        dec4 = self.D4(dec4)                                    # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec4, x3), 1)                         # out: batch * 512 * 64 * 64
        # decoder level 3
        dec3 = self.D3(dec3)                                    # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, x2), 1)                         # out: batch * 256 * 128 * 128
        # decoder level 2
        dec2 = self.D2(dec2)                                    # out: batch * 64 * 128 * 128
        dec1 = torch.cat((dec2, x1), 1)                         # out: batch * 128 * 256 * 256
        dec1 = self.D1(dec1)                                    # out: batch * 64 * 256 * 256
        # decoder level 1
        residual = self.outconv(dec1)                           # out: batch * 2 * 256 * 256

        out = warped_0 - residual
        
        return out, residual

class Combination_Net_without_long(nn.Module):
    def __init__(self, opt):
        super(Combination_Net_without_long, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.out_channels * 8, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E2 = Conv2dLayer(opt.start_channels_comb, opt.start_channels_comb * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E3 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E4 = Conv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E5 = Conv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T2 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T3 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T4 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Decoder
        self.D5 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels_comb * 16, opt.start_channels_comb * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D1 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.outconv = Conv2dLayer(opt.start_channels_comb, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')
        
    def forward(self, p_t, warn_outs):
        
        # Input:
        concat_input = torch.cat((p_t, warn_outs), 1)

        # encoder
        x1 = self.E1(concat_input)                              # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16

        # bottleneck
        x5 = self.T1(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T2(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T3(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T4(x5)                                        # out: batch * 512 * 16 * 16

        # decoder
        # decoder level 5
        dec5 = self.D5(x5)                                      # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec5, x4), 1)                         # out: batch * 1024 * 32 * 32
        # decoder level 4
        dec4 = self.D4(dec4)                                    # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec4, x3), 1)                         # out: batch * 512 * 64 * 64
        # decoder level 3
        dec3 = self.D3(dec3)                                    # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, x2), 1)                         # out: batch * 256 * 128 * 128
        # decoder level 2
        dec2 = self.D2(dec2)                                    # out: batch * 64 * 128 * 128
        dec1 = torch.cat((dec2, x1), 1)                         # out: batch * 128 * 256 * 256
        dec1 = self.D1(dec1)                                    # out: batch * 64 * 256 * 256
        # decoder level 1
        residual = self.outconv(dec1)                           # out: batch * 2 * 256 * 256

        out = p_t - residual
        
        return out, residual

class Combination_Net_without_short_and_long(nn.Module):
    def __init__(self, opt):
        super(Combination_Net_without_short_and_long, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.out_channels, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E2 = Conv2dLayer(opt.start_channels_comb, opt.start_channels_comb * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E3 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E4 = Conv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.E5 = Conv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T2 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T3 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.T4 = ResConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        # Decoder
        self.D5 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels_comb * 16, opt.start_channels_comb * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels_comb * 8, opt.start_channels_comb * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels_comb * 4, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb, scale_factor = 2)
        self.D1 = Conv2dLayer(opt.start_channels_comb * 2, opt.start_channels_comb, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_comb, norm = opt.norm_comb)
        self.outconv = Conv2dLayer(opt.start_channels_comb, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')
        
    def forward(self, p_t):
        
        # encoder
        x1 = self.E1(p_t)                                       # out: batch * 64 * 256 * 256
        x2 = self.E2(x1)                                        # out: batch * 128 * 128 * 128
        x3 = self.E3(x2)                                        # out: batch * 256 * 64 * 64
        x4 = self.E4(x3)                                        # out: batch * 512 * 32 * 32
        x5 = self.E5(x4)                                        # out: batch * 512 * 16 * 16

        # bottleneck
        x5 = self.T1(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T2(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T3(x5)                                        # out: batch * 512 * 16 * 16
        x5 = self.T4(x5)                                        # out: batch * 512 * 16 * 16

        # decoder
        # decoder level 5
        dec5 = self.D5(x5)                                      # out: batch * 512 * 32 * 32
        dec4 = torch.cat((dec5, x4), 1)                         # out: batch * 1024 * 32 * 32
        # decoder level 4
        dec4 = self.D4(dec4)                                    # out: batch * 256 * 64 * 64
        dec3 = torch.cat((dec4, x3), 1)                         # out: batch * 512 * 64 * 64
        # decoder level 3
        dec3 = self.D3(dec3)                                    # out: batch * 128 * 128 * 128
        dec2 = torch.cat((dec3, x2), 1)                         # out: batch * 256 * 128 * 128
        # decoder level 2
        dec2 = self.D2(dec2)                                    # out: batch * 64 * 128 * 128
        dec1 = torch.cat((dec2, x1), 1)                         # out: batch * 128 * 256 * 256
        dec1 = self.D1(dec1)                                    # out: batch * 64 * 256 * 256
        # decoder level 1
        residual = self.outconv(dec1)                           # out: batch * 2 * 256 * 256

        out = p_t - residual
        
        return out, residual

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--start_channels_comb', type = int, default = 64, help = 'combination net channels')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'visible mask channel')
    parser.add_argument('--comb_channels', type = int, default = 16, help = 'start channel for Warp_Artifact_Removal_Net')
    parser.add_argument('--transformer_channels', type = int, default = 32, help = 'start channel for Transformer')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_comb', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_comb', type = str, default = 'in', help = 'normalization type')
    opt = parser.parse_args()

    net = Combination_Net(opt).cuda()

    prev_state = None
    a = torch.randn(1, 2 * 7, 256, 256).cuda()
    fea1 = torch.randn(1, 512, 32, 32).cuda()
    fea2 = torch.randn(1, 512, 16, 16).cuda()
    b, c = net(a, fea1, fea2, prev_state)
    print(b.shape)
    print(c[0].shape, c[1].shape)
