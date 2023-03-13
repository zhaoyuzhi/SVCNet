import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *

#-----------------------------------------
#         Super-resolution module
#-----------------------------------------
class Color_Embedding_SR_Net(nn.Module):
    def __init__(self, opt):
        super(Color_Embedding_SR_Net, self).__init__()
        # sharing parts
        self.begin = Conv2dLayer(opt.out_channels, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = 'none', sn = False)
        self.sr1 = ResidualDenseBlock_5C(opt.start_channels_sr, opt.start_channels_sr // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.sr2 = ResidualDenseBlock_5C(opt.start_channels_sr, opt.start_channels_sr // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.sr3 = ResidualDenseBlock_5C(opt.start_channels_sr, opt.start_channels_sr // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.sr4 = ResidualDenseBlock_5C(opt.start_channels_sr, opt.start_channels_sr // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.mid = Conv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        # unsharing parts
        self.down2_up1 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down2_final = Conv2dLayer(opt.start_channels_sr, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = False)
        self.down4_up1 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down4_up2 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down4_final = Conv2dLayer(opt.start_channels_sr, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = False)
        self.down8_up1 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down8_up2 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down8_up3 = TransposeConv2dLayer(opt.start_channels_sr, opt.start_channels_sr, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_sr, norm = opt.norm_sr, sn = False)
        self.down8_final = Conv2dLayer(opt.start_channels_sr, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = False)
        
    def forward(self, x):
        # save residual
        residual_up2 = F.interpolate(x, scale_factor = 2, mode = 'nearest')
        residual_up4 = F.interpolate(x, scale_factor = 4, mode = 'nearest')
        residual_up8 = F.interpolate(x, scale_factor = 8, mode = 'nearest')
        # sharing parts
        x = self.begin(x)                                                   # out: batch * 32 * H * W
        x = self.sr1(x)                                                     # out: batch * 32 * H * W
        x = self.sr2(x)                                                     # out: batch * 32 * H * W
        x = self.sr3(x)                                                     # out: batch * 32 * H * W
        x = self.sr4(x)                                                     # out: batch * 32 * H * W
        x = self.mid(x)                                                     # out: batch * 32 * H * W
        # unsharing parts
        x_up2 = self.down2_up1(x)
        x_up2 = self.down2_final(x_up2)

        x_up4 = self.down4_up1(x)
        x_up4 = self.down4_up2(x_up4)
        x_up4 = self.down4_final(x_up4)

        x_up8 = self.down8_up1(x)
        x_up8 = self.down8_up2(x_up8)
        x_up8 = self.down8_up3(x_up8)
        x_up8 = self.down8_final(x_up8)
        # residual learning
        out_from_up2 = residual_up2 - x_up2
        out_from_up2 = torch.clamp(out_from_up2, 0, 1)
        out_from_up4 = residual_up4 - x_up4
        out_from_up4 = torch.clamp(out_from_up4, 0, 1)
        out_from_up8 = residual_up8 - x_up8
        out_from_up8 = torch.clamp(out_from_up8, 0, 1)
        return out_from_up2, out_from_up4, out_from_up8


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'visible mask channel')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--start_channels_warn', type = int, default = 16, help = 'warn channel')
    parser.add_argument('--start_channels_comb', type = int, default = 64, help = 'combination net channels')
    parser.add_argument('--start_channels_sr', type = int, default = 32, help = 'super resolution net channel')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_warn', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_comb', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_sr', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_warn', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm_comb', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_sr', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    opt = parser.parse_args()

    net = Color_Embedding_SR_Net(opt).cuda()
    a = torch.randn(1, 2, 128, 256).cuda()
    s = torch.randn(1, 1, 128, 256).cuda()
    b = net(a, s)
    print(b.shape)
