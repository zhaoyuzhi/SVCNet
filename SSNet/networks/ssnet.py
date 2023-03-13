import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *
from networks.ssnet_warn import Warp_Artifact_Removal_Net
from networks.ssnet_corr import Correspondence_Net
from networks.ssnet_comb import Combination_Net, Combination_Net_without_short, Combination_Net_without_long, Combination_Net_without_short_and_long
from networks.ssnet_sr import Color_Embedding_SR_Net
import networks.network_utils as nutils

# ----------------------------------------
#           SSNet (full model)
# ----------------------------------------
class SSNet(nn.Module):
    def __init__(self, opt):
        super(SSNet, self).__init__()

        self.warn = Warp_Artifact_Removal_Net(opt)
        self.corr = Correspondence_Net(opt)
        for p in self.warn.parameters():
            p.requires_grad = False
        for p in self.corr.parameters():
            p.requires_grad = False

        self.comb = Combination_Net(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, lab_0, feature_0, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # warn
        for i in range(len(p_t_list)):
            # finetune cpnet outputs
            refined = self.warn(p_t_list[i], mask_flow_list[i])
            if i == 0:
                warn_outs = refined
            else:
                warn_outs = torch.cat((warn_outs, refined), 1)
            # finetune last ssnet output when using t-1 optical flow
            if i == 2:
                refined = self.warn(last_out, mask_flow_list[i])
                warn_outs = torch.cat((warn_outs, refined), 1)

        # corr
        warped_0, similarity_map = self.corr(x_t, lab_0, feature_0, temperature = 0.01)

        # output
        out, residual = self.comb(p_t, warped_0, warn_outs)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up2, residual

# ----------------------------------------
#       SSNet (for ablation study)
# ----------------------------------------
class SSNet_without_short(nn.Module):
    def __init__(self, opt):
        super(SSNet_without_short, self).__init__()

        self.corr = Correspondence_Net(opt)
        for p in self.corr.parameters():
            p.requires_grad = False

        self.comb = Combination_Net_without_short(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, lab_0, feature_0, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # corr
        warped_0, similarity_map = self.corr(x_t, lab_0, feature_0, temperature = 0.01)

        # output
        out, residual = self.comb(p_t, warped_0)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up2, residual

class SSNet_without_long(nn.Module):
    def __init__(self, opt):
        super(SSNet_without_long, self).__init__()

        self.warn = Warp_Artifact_Removal_Net(opt)
        for p in self.warn.parameters():
            p.requires_grad = False

        self.comb = Combination_Net_without_long(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # warn
        for i in range(len(p_t_list)):
            # finetune cpnet outputs
            refined = self.warn(p_t_list[i], mask_flow_list[i])
            if i == 0:
                warn_outs = refined
            else:
                warn_outs = torch.cat((warn_outs, refined), 1)
            # finetune last ssnet output when using t-1 optical flow
            if i == 2:
                refined = self.warn(last_out, mask_flow_list[i])
                warn_outs = torch.cat((warn_outs, refined), 1)

        # output
        out, residual = self.comb(p_t, warn_outs)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up2, residual

class SSNet_without_short_and_long(nn.Module):
    def __init__(self, opt):
        super(SSNet_without_short_and_long, self).__init__()

        self.comb = Combination_Net_without_short_and_long(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # output
        out, residual = self.comb(p_t)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up2, residual

class SSNet_64p(nn.Module):
    def __init__(self, opt):
        super(SSNet_64p, self).__init__()

        self.warn = Warp_Artifact_Removal_Net(opt)
        self.corr = Correspondence_Net(opt)
        for p in self.warn.parameters():
            p.requires_grad = False
        for p in self.corr.parameters():
            p.requires_grad = False

        self.comb = Combination_Net(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, lab_0, feature_0, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # warn
        for i in range(len(p_t_list)):
            # finetune cpnet outputs
            refined = self.warn(p_t_list[i], mask_flow_list[i])
            if i == 0:
                warn_outs = refined
            else:
                warn_outs = torch.cat((warn_outs, refined), 1)
            # finetune last ssnet output when using t-1 optical flow
            if i == 2:
                refined = self.warn(last_out, mask_flow_list[i])
                warn_outs = torch.cat((warn_outs, refined), 1)

        # corr
        warped_0, similarity_map = self.corr(x_t, lab_0, feature_0, temperature = 0.01)

        # output
        out, residual = self.comb(p_t, warped_0, warn_outs)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up8, residual

class SSNet_128p(nn.Module):
    def __init__(self, opt):
        super(SSNet_128p, self).__init__()

        self.warn = Warp_Artifact_Removal_Net(opt)
        self.corr = Correspondence_Net(opt)
        for p in self.warn.parameters():
            p.requires_grad = False
        for p in self.corr.parameters():
            p.requires_grad = False

        self.comb = Combination_Net(opt)
        self.srnet = Color_Embedding_SR_Net(opt)
        
    def forward(self, x_t, lab_0, feature_0, p_t, last_out, p_t_list, mask_flow_list):
        # Input:
        # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
        # lab_0: the first colorized frame (CIE Lab) by CPNet
        # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
        # p_t: the current colorized frame (t) by CPNet
        # last_out: the warped last output by SSNet
        # p_t_list: the warped colorized frames by CPNet
        # mask_flow_list: masks of optical flow

        # warn
        for i in range(len(p_t_list)):
            # finetune cpnet outputs
            refined = self.warn(p_t_list[i], mask_flow_list[i])
            if i == 0:
                warn_outs = refined
            else:
                warn_outs = torch.cat((warn_outs, refined), 1)
            # finetune last ssnet output when using t-1 optical flow
            if i == 2:
                refined = self.warn(last_out, mask_flow_list[i])
                warn_outs = torch.cat((warn_outs, refined), 1)

        # corr
        warped_0, similarity_map = self.corr(x_t, lab_0, feature_0, temperature = 0.01)

        # output
        out, residual = self.comb(p_t, warped_0, warn_outs)

        # sr
        out_from_up2, out_from_up4, out_from_up8 = self.srnet(out)

        return out, out_from_up4, residual

if __name__ == "__main__":

    import argparse
    import cv2
    from PIL import Image
    import numpy as np
    import datasets.data_utils as dutils

    parser = argparse.ArgumentParser()
    # Pre-trained model parameters
    parser.add_argument('--cpnet_path', type = str, default = '../trained_models/CPNet/models_2nd_vimeo_64p/CPNet_VGG16_Seg/cpnet_epoch10_batchsize32.pth', help = 'the load name of models')
    parser.add_argument('--ssnet_path', type = str, default = '', help = 'the load name of models')
    parser.add_argument('--warn_path', type = str, default = '../trained_models/WARN/Warp_Artifact_Removal_Net_256p_in_epoch2000_bs16.pth', help = 'the load name of models')
    parser.add_argument('--corrnet_vgg_path', type = str, default = '../trained_models/CorrNet/vgg19_conv.pth', help = 'the load name of models')
    parser.add_argument('--corrnet_nonlocal_path', type = str, default = '../trained_models/CorrNet/nonlocal_net_iter_76000.pth', help = 'the load name of models')
    parser.add_argument('--srnet_path', type = str, default = '../trained_models/SRNet/Color_Embedding_SR_Net_normnone_epoch40_bs4.pth', help = 'the load name of models')
    parser.add_argument('--pwcnet_path', type = str, default = '../trained_models/pwcNet-default.pytorch', help = 'the load name of models')
    parser.add_argument('--perceptual_path', type = str, default = '../trained_models/vgg16_pretrained.pth', help = 'the load name of models')
    parser.add_argument('--vgg_name', type = str, default = '../trained_models/vgg16_pretrained.pth', help = 'the load name of models')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'visible mask channel')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--start_channels_warn', type = int, default = 16, help = 'warn channel')
    parser.add_argument('--start_channels_comb', type = int, default = 32, help = 'combination net channels')
    parser.add_argument('--start_channels_sr', type = int, default = 32, help = 'super resolution net channel')
    parser.add_argument('--lambda_value', type = float, default = 500, help = 'lambda_value of WLS')
    parser.add_argument('--sigma_color', type = float, default = 4, help = 'sigma_color of WLS')
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
    # Dataset parameters
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size') # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument('--crop_size_w', type = int, default = 448, help = 'single patch size') # second stage (128p, 256p, 448p): 256, 448, 832
    opt = parser.parse_args()

    transform = dutils.create_transform(opt)

    def load_dict(process_net, pretrained_net):
        # Get the dict from pre-trained network
        pretrained_dict = pretrained_net
        # Get the dict from processing network
        process_dict = process_net.state_dict()
        # Delete the extra keys of pretrained_dict that do not belong to process_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
        # Update process_dict using pretrained_dict
        process_dict.update(pretrained_dict)
        # Load the updated dict to processing network
        process_net.load_state_dict(process_dict)
        return process_net

    def get_lab(imgpath, opt):
        # Pre-processing, let all the images are in RGB color space
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)                                             # read one image (RGB)
        img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        # Convert RGB to Lab, finally get Tensor
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)                      # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img_l = lab[:, :, [0]]
        img_l = np.concatenate((img_l, img_l, img_l), axis = 2)
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        # Normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
        return img_l, img_ab

    def get_transform(imgpath):
        img_transform = Image.open(imgpath)
        if img_transform.mode == 'L':
            img_transform = img_transform.convert('RGB')
        img_transform = transform(img_transform)                  # 3 * H * W
        img_transform = img_transform[[0], :, :].unsqueeze(0)
        return img_transform

    ssnet = SSNet(opt).cuda()
    if opt.warn_path:
        pretrained_net = torch.load(opt.warn_path)
        ssnet.warn = load_dict(ssnet.warn, pretrained_net)
        print('succ')
    if opt.corrnet_vgg_path:
        pretrained_net = torch.load(opt.corrnet_vgg_path)
        ssnet.corr.vggnet = load_dict(ssnet.corr.vggnet, pretrained_net)
        print('succ')
    if opt.corrnet_nonlocal_path:
        pretrained_net = torch.load(opt.corrnet_nonlocal_path)
        ssnet.corr.nonlocal_net = load_dict(ssnet.corr.nonlocal_net, pretrained_net)
        print('succ')
    if opt.srnet_path:
        pretrained_net = torch.load(opt.srnet_path)
        ssnet.srnet = load_dict(ssnet.srnet, pretrained_net)
        print('succ')

    # x_t: the grayscale of current time, note that it should be transformed to match the requirement of CorrNet
    # lab_0: the first colorized frame (CIE Lab) by CPNet
    # feature_0: the VGG features of the first colorized frame (RGB) by CPNet
    # p_t: the current colorized frame (t) by CPNet
    # last_out: the warped last output by SSNet
    # p_t_list: the warped colorized frames by CPNet
    # mask_flow_list: masks of optical flow
    '''
    x_0 = torch.randn(1, 1, 256, 448).cuda()
    x_t = torch.randn(1, 1, 256, 448).cuda()
    p_0 = torch.randn(1, 2, 256, 448).cuda()
    p_t = torch.randn(1, 2, 256, 448).cuda()
    last_out = torch.randn(1, 2, 256, 448).cuda()
    p_t_list = []
    mask_flow_list = []
    p_t_list_temp = torch.randn(1, 2, 256, 448).cuda()
    mask_flow_list_temp = torch.randn(1, 1, 256, 448).cuda()
    for _ in range(7):
        p_t_list.append(p_t_list_temp)
        mask_flow_list.append(mask_flow_list_temp)

    p_0_lab = dutils.cpnet_ab_to_PIL_rgb(x_0, p_0)[0]
    IB_lab_large = transform(p_0_lab).unsqueeze(0).cuda()
    IB_lab = F.interpolate(IB_lab_large, scale_factor = 0.5, mode = "bilinear")
    with torch.no_grad():
        I_reference_l = IB_lab[:, 0:1, :, :]
        I_reference_ab = IB_lab[:, 1:3, :, :]
        I_reference_rgb = dutils.tensor_lab2rgb(torch.cat((dutils.uncenter_l(I_reference_l), I_reference_ab), dim = 1))
        features_B = ssnet.corr.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess = True)

    b, _ = ssnet(x_t, IB_lab, features_B, p_t, last_out, p_t_list, mask_flow_list)
    print(b.shape)
    '''

    # Define the paths to 0-th frame and t-th frame
    path_to_first_frame = '/home/zyz/Documents/svcnet/SVCNet_comparison_DAVIS_videvo/CIC/DAVIS/bike-packing/00000.jpg'
    path_to_current_frame = '/home/zyz/Documents/svcnet/2dataset_grayscale/DAVIS/bike-packing/00040.jpg'
    #path_to_current_frame = "/home/zyz/Documents/svcnet/2dataset_grayscale/DAVIS/dance-twirl/00050.jpg"

    # Define inputs for the SSNet
    x_t = get_transform(path_to_current_frame)
    x_t = x_t.cuda()
    p_t = torch.randn(1, 2, 256, 448).cuda()
    last_out = torch.randn(1, 2, 256, 448).cuda()
    p_t_list = []
    mask_flow_list = []
    p_t_list_temp = torch.randn(1, 2, 256, 448).cuda()
    mask_flow_list_temp = torch.randn(1, 1, 256, 448).cuda()
    for _ in range(6):
        p_t_list.append(p_t_list_temp)
        mask_flow_list.append(mask_flow_list_temp)

    p_0_lab = transform(Image.open(path_to_first_frame)).unsqueeze(0).cuda()
    #IB_lab = F.interpolate(p_0_lab, scale_factor = 0.5, mode = "bilinear")
    IB_lab = p_0_lab
    with torch.no_grad():
        I_reference_l = IB_lab[:, 0:1, :, :]
        I_reference_ab = IB_lab[:, 1:3, :, :]
        I_reference_rgb = dutils.tensor_lab2rgb(torch.cat((dutils.uncenter_l(I_reference_l), I_reference_ab), dim = 1))
        features_B = ssnet.corr.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess = True)
    
    # Forward SSNet
    b, _, temp_rgb = ssnet(x_t, IB_lab, features_B, p_t, last_out, p_t_list, mask_flow_list)
    
    # Save output images
    temp_rgb = cv2.cvtColor(temp_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('temp.png', temp_rgb) # It should be grayscale of time t and color of time 0
    temp_rgb = cv2.cvtColor(temp_rgb, cv2.COLOR_RGB2BGR)
    
    x_t_cv2_gray, _ = get_lab(path_to_current_frame, opt)
    x_t_cv2_gray = x_t_cv2_gray[:, [0], :, :]
    warped_0_to_cpnet_ab = nutils.cv2_rgb_to_tensor_ab(temp_rgb).cuda()
    warped_0_PIL_rgb = dutils.cpnet_ab_to_PIL_rgb(x_t_cv2_gray, warped_0_to_cpnet_ab)
    warped_0_PIL_rgb = warped_0_PIL_rgb[0]
    IB_lab_large = transform(warped_0_PIL_rgb).unsqueeze(0).cuda()
    #IB_lab = F.interpolate(IB_lab_large, scale_factor = 0.5, mode = "bilinear")
    IB_lab = IB_lab_large
    warped_0_to_cpnet_rgb = nutils.corr_lab_to_cv2_rgb(IB_lab, x_t)
    temp_rgb = cv2.cvtColor(warped_0_to_cpnet_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('temp2.png', temp_rgb)

    '''
    # this part is good
    warped_0_to_cpnet_rgb = nutils.corr_lab_to_cv2_rgb(IB_lab, x_0)
    warped_0_to_cpnet_rgb = cv2.cvtColor(warped_0_to_cpnet_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('temp3.png', warped_0_to_cpnet_rgb)
    '''
