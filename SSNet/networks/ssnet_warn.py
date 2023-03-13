import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *

# ----------------------------------------
#          WARN for Lab embeddings
# ----------------------------------------
class Warp_Artifact_Removal_Net(nn.Module):
    def __init__(self, opt):
        super(Warp_Artifact_Removal_Net, self).__init__()
        self.down1 = Conv2dLayer(opt.out_channels + opt.mask_channels, opt.start_channels_warn, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = 'none', sn = False)
        self.down2 = Conv2dLayer(opt.start_channels_warn, opt.start_channels_warn * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False)
        self.down3 = Conv2dLayer(opt.start_channels_warn * 2, opt.start_channels_warn * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False)
        self.bottleneck = nn.Sequential(
            ResConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False),
            ResConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False),
            ResConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False),
            ResConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False),
        )
        self.up1 = TransposeConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False, scale_factor = 2)
        self.up2 = TransposeConv2dLayer(opt.start_channels_warn * 4, opt.start_channels_warn, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_warn, norm = opt.norm_warn, sn = False, scale_factor = 2)
        self.up3 = Conv2dLayer(opt.start_channels_warn * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'tanh', norm = 'none', sn = False)
        
    def forward(self, x, mask):
        # save residual
        residual = x
        # forward
        x = torch.cat((x, mask), 1)                                 # out: batch * (2+1) * 256 * 256
        d1 = self.down1(x)                                          # out: batch * 16 * 256 * 256
        d2 = self.down2(d1)                                         # out: batch * 32 * 128 * 128
        d3 = self.down3(d2)                                         # out: batch * 64 * 64 * 64
        d3 = self.bottleneck(d3)                                    # out: batch * 64 * 64 * 64
        up1 = self.up1(d3)                                          # out: batch * 32 * 128 * 128
        up1 = torch.cat((up1, d2), 1)                               # out: batch * (32+32) * 128 * 128
        up2 = self.up2(up1)                                         # out: batch * 16 * 256 * 256
        up2 = torch.cat((up2, d1), 1)                               # out: batch * (16+16) * 256 * 256
        up3 = self.up3(up2)                                         # out: batch * 2 * 256 * 256
        # residual learning
        out = residual - up3
        return out


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

    net = Warp_Artifact_Removal_Net(opt).cuda()
    a = torch.randn(1, 2, 128, 256).cuda()
    s = torch.randn(1, 1, 128, 256).cuda()
    b = net(a, s)
    print(b.shape)
