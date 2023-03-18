
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os, PIL

BatchNorm2d = BatchNorm2d_class = nn.BatchNorm2d
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
relu_inplace = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
                'BASIC': BasicBlock,
                'BOTTLENECK': Bottleneck
            }

class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        self.cfg = config
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x_fp = x

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        if self.cfg.MODEL.NAME == "seg_hrnet_mpc":
            # import pdb; pdb.set_trace()
            return x, x_fp, None
        return x

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()              
            # pretrained_dict = {k: v for k, v in pretrained_dict.items()
            #                    if k in model_dict.keys()}
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                                if k[6:] in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class InstancePyramid():
    inst_count = 0
    def __init__(self, pos, init_lvl, lvl_sizes, ins_tar_id=-1, tar_cat=-1):
        self.idx = InstancePyramid.inst_count
        InstancePyramid.inst_count += 1
        self.pos = pos  # tuple required
        self.init_lvl = init_lvl
        self.lvl_sizes = lvl_sizes
        self.init_size = self.lvl_sizes[self.init_lvl]
        self.ins_tar_id = ins_tar_id
        self.tar_cat = tar_cat
        self.mask_129 = None
        self.masks = {}

        init_mask = torch.zeros(self.init_size)
        init_mask[self.pos] = 1
        cover_conv = nn.Conv2d(1,1,3,1,1, bias=False)
        cover_conv.weight.requires_grad = False
        cover_conv.weight.data.fill_(1)
        self.init_cover = cover_conv(init_mask[None,None])
        self.init_center_cover = cover_conv(init_mask[None,None])
        # center_mask = torch.ones(3,3)
        # if self.pos[0] <= 0:
        #     center_mask[:1, :] = 0
        # elif self.pos[0] >= self.init_size[0] - 1:
        #     center_mask[-1:, :] = 0
        # if self.pos[1] <= 0:
        #     center_mask[:, :1] = 0
        # elif self.pos[1] >= self.init_size[1] - 1:
        #     center_mask[:, -1:] = 0
        # self.center_mask = center_mask[None,None]

    def get_cover_mask(self, lvl):
        f_mask = F.interpolate(self.init_cover, self.lvl_sizes[lvl], mode='nearest')[0,0]
        # import pdb; pdb.set_trace()
        positive_pos = torch.nonzero(f_mask)
        pos_range = (positive_pos[:,0].min().item(), positive_pos[:,0].max().item()+1,
                    positive_pos[:,1].min().item(), positive_pos[:,1].max().item()+1)
        return f_mask, pos_range
        
    def set_mask(self, lvl, mask):
        self.masks[lvl] = mask

    def get_mask(self, lvl):
        return self.masks[lvl]

    def get_rel_pos(self, pub_level):
        init_size = self.lvl_sizes[self.init_lvl]
        req_size = self.lvl_sizes[pub_level]

        h = round((self.pos[0]+0.5) / init_size[0] * req_size[0]-0.5)
        w = round((self.pos[1]+0.5) / init_size[1] * req_size[1]-0.5)

        return (h, w)

class KeepDown(nn.Module):
    def __init__(self, cfg):
        super(KeepDown, self).__init__()
        self.down5 = BasicBlock(384, 512, stride=2, downsample=nn.Conv2d(384, 512, 1, 2))
        self.down6 = BasicBlock(512, 1024, stride=2, downsample=nn.Conv2d(512, 1024, 1, 2))
        self.down7 = BasicBlock(1024, 1024, stride=2, downsample=nn.Conv2d(1024, 1024, 1, 2))
        # self.down8 = BasicBlock(1024, 1024, stride=2, downsample=nn.Conv2d(1024, 1024, 1, 2))
        # self.down9 = BasicBlock(1024, 1024, stride=2, downsample=nn.Conv2d(1024, 1024, 1, 2))

    def forward(self, x2_to_5):
        x5= x2_to_5[-1]
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        # x9 = self.down8(x8)

        return x2_to_5+[x6, x7, x8]
        # return [x6, x7, x8, x9]

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UpConv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # self.conv = DoubleConv3x3(in_ch, out_ch)
        self.conv = BasicBlock(in_ch, out_ch, downsample=nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        # x = torch.cat([x2, x1], dim=1)
        x = x1 + x2
        x = self.conv(x)
        return x

class CategoryPyramid(nn.Module):
    def __init__(self, cfg):
        super(CategoryPyramid, self).__init__()
        inchns = [1024, 1024, 512, 384, 192, 96, 48]
        self.f_down_1x1 = nn.ModuleList([nn.ModuleList([nn.Conv2d(inchns[j],512,1) for \
            i in range(j+1)]) for j in range(6)])
        self.branches = nn.ModuleList([nn.Sequential(
            BasicBlock(512, 512),
            # BasicBlock(512, 512),
            # BasicBlock(512, 512),
            # BasicBlock(512, 512),
        ) for j in range(7)])

        self.f_down_1x1_128 = nn.ModuleList([nn.Conv2d(48,512,1) for j in range(6)])
        self.up1 = UpConv(512, 512)    # 2
        self.up2 = UpConv(512, 512)    # 4
        self.up3 = UpConv(512, 512)    # 8
        self.up4 = UpConv(512, 512)    # 16
        self.up5 = UpConv(512, 512)    # 32
        self.up6 = UpConv(512, 512)    # 64
        self.up128 = UpConv(512, 256)    # 3
        
        self.up_list = [None, self.up1, self.up2, self.up3, self.up4, self.up5, self.up6]
        # self.up_list = nn.ModuleList([None, self.up1, self.up2, self.up3, 
        #     self.up4, self.up5, self.up6])

        self.last_layer = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(128, 8, kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL, padding=1 \
                if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0)
        )

class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        self.s_show_iter = False
        self.color_bias = 5
        num_classes = cfg.DATASET.NUM_CLASSES
        self.sematic_bone = HighResolutionNet(cfg)
        if self.cfg.MODEL.PRETRAINED:
            self.sematic_bone.init_weights(self.cfg.MODEL.PRETRAINED)
        
        self.keep_down = KeepDown(cfg)
        self.cat_conv = CategoryPyramid(cfg)
        self.sematic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.TRAIN.IGNORE_LABEL)
        self.panoptic_criteron = nn.CrossEntropyLoss(ignore_index=cfg.TRAIN.IGNORE_LABEL)

    def load_weights(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()              
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image, targets=None):
        InstancePyramid.inst_count = 0
        target_assistant = False
        output_dict = dict()
        N, _, img_h, img_w = image.shape
        sematic_info = self.sematic_bone(image)
        sem_mask_128, down_features, up_features = sematic_info
        sematic_out = F.interpolate(sem_mask_128, scale_factor=4, mode='bilinear', align_corners=True)
        # sematic_info = (F.interpolate(sematic_info[0], scale_factor=4, mode='bilinear', align_corners=True),) + sematic_info[1:]
        # sematic_out = sematic_info[0]
        sematic_loss = self.sematic_criteron(sematic_out, targets['label'].long())
        sematic_info = (sematic_out, down_features, up_features, sematic_loss,)
        # sematic_out, down_features, up_features, sematic_loss = sematic_info

        output_dict['loss_dict'] = {'sematic': sematic_loss.mean(), 'instance': 0}
        output_dict['sematic_out'] = sematic_out
        output_dict['targets'] = targets
        output_dict['pano_target'] = []
        output_dict['ins_pyramids'] = [[] for _ in range(N)]
        output_dict['observe_images'] = [[] for _ in range(N)]
        
        # keep_down_features = self.keep_down(down_features)

        if self.cfg.TRAIN.SEMATIC_ONLY:
            if self.s_show_iter:
                output_dict['show_img'] = self.make_observation(image, sematic_out=sematic_out, 
                        sem_targets=targets['label'], ins_target=targets['instance'], 
                        s_sem_tar=True, s_ins_tar=True, s_sem_out=True, 
                        label_of_insIds=targets['label_of_insIds'], )
                # self.show_image(image, output_dict, show_via='plt')
            return output_dict