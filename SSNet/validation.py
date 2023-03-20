import argparse
import os
import cv2
from PIL import Image
import numpy as np
import torch

import datasets.data_utils as dutils
import networks.pwcnet as pwcnet
import utils

def define_dataset(opt):
    # Inference for color scribbles
    imglist = utils.text_readlines(os.path.join(opt.txt_root, opt.tag + '_test_imagelist.txt'))
    classlist = utils.text_readlines(os.path.join(opt.txt_root, opt.tag + '_test_class.txt'))
    imgroot = [list() for i in range(len(classlist))]

    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(imglist):
            if imgname.split('/')[-2] == classname:
                imgroot[i].append(imgname)

    print('There are %d videos in the test set.' % (len(imgroot)))
    return imgroot

def recover_ndarray_to_tensor(img):
    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return img

def convert_lab_to_bgr(input_img, out_img):
    input_img = input_img[0, [0], :, :].data.cpu().numpy().transpose(1, 2, 0)           # 256 * 256 * 1
    out_img = out_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                 # 256 * 256 * 2
    # Resize the output ab embedding if the size is not like the input L component
    if input_img.shape[0] != out_img.shape[0] or input_img.shape[1] != out_img.shape[1]:
        out_img = cv2.resize(out_img, (input_img.shape[1], input_img.shape[0]))
    out_img = np.concatenate((input_img, out_img), axis = 2)                            # 256 * 256 * 3
    out_img = (out_img * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_Lab2BGR)                                  # 256 * 256 * 3 (√)
    return out_img

def read_img_path(imgroot, i, opt):
    # Read images, opt.iter_frames successive images
    img_paths = []
    scribble_paths = []
    # Get the full paths
    tail = '_%dp.png' % (opt.crop_size_h)
    for j in range(opt.iter_frames):
        img_paths.append(os.path.join(opt.base_root, opt.tag, imgroot[index][i+j].split('/')[0], imgroot[index][i+j].split('/')[1]))
        scribble_paths.append(os.path.join(opt.scribble_root, opt.tag, imgroot[index][i+j].split('/')[0], imgroot[index][i+j].split('/')[1]).replace('.jpg', tail))
    return img_paths, scribble_paths

def read_img(path, opt):
    # Read images
    img = Image.open(path).convert('RGB')
    img = np.array(img) # in RGB format
    img_original = img.copy()
    img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)

    # Input grayscale L and ground truth ab
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    img_l = lab[:, :, [0]]
    img_l = np.concatenate((img_l, img_l, img_l), axis = 2)
    img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
    img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

    # Original size grayscale L and ground truth RGB
    gt_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    gt_bgr = torch.from_numpy(gt_bgr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    img_original_l = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    img_original_l = torch.from_numpy(img_original_l.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous()
    
    return img_l, img_ab, gt_bgr, img_original_l

def read_img_2(path, opt):
    transform_from_PIL = dutils.create_transform(opt)
    img_l_transform = transform_from_PIL(Image.open(path))[[0], :, :]
    img_l_transform = img_l_transform.unsqueeze(0)
    return img_l_transform

def read_imgs(img_paths, scribble_paths, opt):
    # Read images, opt.iter_frames successive images
    img_lists = []
    img_transform_lists = []
    scribble_lists = []
    gt_ab_lists = []
    gt_bgr_lists = []
    img_original_lists = []

    for j in range(opt.iter_frames):
        # Read images and scribbles one-by-one
        img, gt_ab, gt_bgr, img_original_l = read_img(img_paths[j], opt)
        img_transform = read_img_2(img_paths[j], opt)
        _, scribble, _, _ = read_img(scribble_paths[j], opt)
        
        # If not using given color scribbles, let them equal 0
        if not opt.use_scribble:
            scribble = scribble.fill_(0)

        # To device
        img = img.cuda()
        img_transform = img_transform.cuda()
        scribble = scribble.cuda()
        gt_ab = gt_ab.cuda()
        gt_bgr = gt_bgr.cuda()
        img_original_l = img_original_l.cuda()

        img_lists.append(img)
        img_transform_lists.append(img_transform)
        scribble_lists.append(scribble)
        gt_ab_lists.append(gt_ab)
        gt_bgr_lists.append(gt_bgr)
        img_original_lists.append(img_original_l)

    return img_lists, img_transform_lists, scribble_lists, gt_ab_lists, gt_bgr_lists, img_original_lists

def save_img(tensor_L, tensor_ab, save_name, opt):
    bgr = convert_lab_to_bgr(tensor_L, tensor_ab)
    save_rgb_sub_folder_name = os.path.join(opt.save_rgb_path, opt.tag, save_name.split('/')[0])
    utils.check_path(save_rgb_sub_folder_name)
    save_rgb_name = os.path.join(save_rgb_sub_folder_name, save_name.split('/')[1].replace('.jpg', '.png'))
    cv2.imwrite(save_rgb_name, bgr)
    return bgr

def compute_metrics(bgr, gt_bgr):
    bgr = recover_ndarray_to_tensor(bgr).cuda()
    assert bgr.shape == gt_bgr.shape
    this_PSNR = utils.psnr(bgr, gt_bgr, 1) * gt_bgr.shape[0]
    this_SSIM = utils.ssim(bgr, gt_bgr) * gt_bgr.shape[0]
    return this_PSNR, this_SSIM

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train_cpnet_type', type = str, default = 'CPNet_VGG16_Seg', help = 'pre_train_cpnet_type')
    parser.add_argument('--pre_train_ssnet_type', type = str, default = 'SSNet', help = 'pre_train_ssnet_type')
    parser.add_argument('--tag', type = str, default = 'DAVIS', help = 'DAVIS | videvo')
    parser.add_argument('--save_rgb_path', type = str, \
        default = './val_result', \
            help = 'save the generated rgb image to certain path')
    parser.add_argument('--cpnet_path', type = str, default = './trained_models/CPNet/models_2nd_dv_256p/CPNet_VGG16_Seg/cpnet_epoch1000_batchsize32.pth', help = 'the load name of models')
    parser.add_argument('--ssnet_path', type = str, default = './trained_ssnet/SSNet/ssnet_epoch2000_bs8.pth', help = 'the load name of models')
    parser.add_argument('--pwcnet_path', type = str, default = './trained_models/PWCNet/pwcNet-default.pytorch', help = 'the load name of models')
    parser.add_argument('--vgg_name', type = str, default = './trained_models/Others/vgg16_pretrained.pth', help = 'the load name of models')
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
    # loss balancing parameters
    parser.add_argument('--mask_para', type = float, default = 200, help = 'coefficient for visible mask')
    # Dataset parameters
    parser.add_argument('--base_root', type = str, \
        default = './data/DAVIS_Videvo/val', \
            help = 'the base validation folder')
    parser.add_argument('--scribble_root', type = str, \
        default = './color_point40_color_width5_256p', \
            help = 'the base validation folder')
    parser.add_argument('--txt_root', type = str, default = './txt', help = 'the base training folder')
    parser.add_argument('--iter_frames', type = int, default = 7, help = 'number of iter_frames in one iteration; +1 since the first frame is not counted')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size') # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument('--crop_size_w', type = int, default = 448, help = 'single patch size') # second stage (128p, 256p, 448p): 256, 448, 832
    parser.add_argument('--use_scribble', type = bool, default = True, help = 'the flag to use given color scribbles')
    opt = parser.parse_args()
    #print(opt)
    
    transform_from_PIL = dutils.create_transform(opt)

    # ----------------------------------------
    #       Initialize testing network
    # ----------------------------------------

    # Initialize Generator
    cpnet, ssnet = utils.create_generator(opt, tag = 'val')
    flownet = utils.create_pwcnet(opt)

    # To device
    cpnet = cpnet.cuda()
    ssnet = ssnet.cuda()
    flownet = flownet.cuda()
    
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    imgroot = define_dataset(opt)
    
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    
    # Define whether using the long-range connection
    use_long_connection = False
    if hasattr(ssnet, 'module'):
        if hasattr(ssnet.module, 'corr'):
            use_long_connection = True
    else:
        if hasattr(ssnet, 'corr'):
            use_long_connection = True

    # forward
    val_PSNR = 0
    val_SSIM = 0
    count = 0
    
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):
        
        N = len(imgroot[index])

        for i in range(0, N - opt.iter_frames + 1):
            # Define paths, the output should contain opt.iter_frames paths
            img_paths, scribble_paths = read_img_path(imgroot, i, opt)
            
            # Read images
            input_frames, input_transform_frames, scribble_frames, gt_ab_frames, gt_bgr_frames, img_original_l = read_imgs(img_paths, scribble_paths, opt)
            
            # Forward CPNet first to obtain the colorized frames
            cpnet_frames = []
            with torch.no_grad():
                for j in range(opt.iter_frames):
                    x_t = input_frames[j][:, [0], :, :]
                    color_scribble = scribble_frames[j]
                    cpnet_out, _ = cpnet(x_t, color_scribble)
                    cpnet_frames.append(cpnet_out)
            # save the first CPNet's output
            #cpnet_out_0 = cpnet_frames[0]

            # Forward SSNet then
            with torch.no_grad():
                # warp previous and leading frames through optical flows
                center_id = (opt.iter_frames - 1) // 2
                flow_minus3_to_current = pwcnet.PWCEstimate(flownet, input_frames[0], input_frames[center_id], drange = True, reshape = True)
                flow_minus2_to_current = pwcnet.PWCEstimate(flownet, input_frames[1], input_frames[center_id], drange = True, reshape = True)
                flow_minus1_to_current = pwcnet.PWCEstimate(flownet, input_frames[2], input_frames[center_id], drange = True, reshape = True)
                flow_add1_to_current = pwcnet.PWCEstimate(flownet, input_frames[4], input_frames[center_id], drange = True, reshape = True)
                flow_add2_to_current = pwcnet.PWCEstimate(flownet, input_frames[5], input_frames[center_id], drange = True, reshape = True)
                flow_add3_to_current = pwcnet.PWCEstimate(flownet, input_frames[6], input_frames[center_id], drange = True, reshape = True)
                
                # compute visible mask
                x_t_minus3_warp = pwcnet.PWCNetBackward(input_frames[0], flow_minus3_to_current)
                x_t_minus2_warp = pwcnet.PWCNetBackward(input_frames[1], flow_minus2_to_current)
                x_t_minus1_warp = pwcnet.PWCNetBackward(input_frames[2], flow_minus1_to_current)
                x_t_add1_warp = pwcnet.PWCNetBackward(input_frames[4], flow_add1_to_current)
                x_t_add2_warp = pwcnet.PWCNetBackward(input_frames[5], flow_add2_to_current)
                x_t_add3_warp = pwcnet.PWCNetBackward(input_frames[6], flow_add3_to_current)
                mask_minus3_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus3_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_minus2_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus2_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_minus1_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus1_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add1_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add1_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add2_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add2_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add3_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add3_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)

                # warp CPNet outputs to current position
                cpnet_t_minus3_warp = pwcnet.PWCNetBackward(cpnet_frames[0], flow_minus3_to_current)
                cpnet_t_minus2_warp = pwcnet.PWCNetBackward(cpnet_frames[1], flow_minus2_to_current)
                cpnet_t_minus1_warp = pwcnet.PWCNetBackward(cpnet_frames[2], flow_minus1_to_current)
                cpnet_t_add1_warp = pwcnet.PWCNetBackward(cpnet_frames[4], flow_add1_to_current)
                cpnet_t_add2_warp = pwcnet.PWCNetBackward(cpnet_frames[5], flow_add2_to_current)
                cpnet_t_add3_warp = pwcnet.PWCNetBackward(cpnet_frames[6], flow_add3_to_current)
                
                # save SSNet's last output and warp it
                if i > 0:
                    ssnet_t_minus1 = ssnet_t.detach()
                    ssnet_t_minus1_warp = pwcnet.PWCNetBackward(ssnet_t_minus1, flow_minus1_to_current)
                else:
                    ssnet_t_minus1_warp = cpnet_frames[center_id]
                
                # arrange all inputs for SSNet
                # x_t = input_frames[center_id]
                # cpnet_t = cpnet_frames[center_id]
                if use_long_connection and i == 0:
                    # cpnet_ab_to_PIL_rgb recieves: cv2 format grayscale tensor + cpnet format tensor
                    cpnet_t_0_PIL_rgb = dutils.cpnet_ab_to_PIL_rgb(input_frames[0][:, [0], :, :], cpnet_frames[0])
                    for batch in range(len(cpnet_t_0_PIL_rgb)):
                        cpnet_t_0_PIL_rgb_batch = cpnet_t_0_PIL_rgb[batch]
                        cpnet_t_0_PIL_rgb_batch = transform_from_PIL(cpnet_t_0_PIL_rgb_batch).unsqueeze(0).cuda()
                        if batch == 0:
                            IB_lab = cpnet_t_0_PIL_rgb_batch
                        else:
                            IB_lab = torch.cat((IB_lab, cpnet_t_0_PIL_rgb_batch), 0)
                    I_reference_l = IB_lab[:, 0:1, :, :]
                    I_reference_ab = IB_lab[:, 1:3, :, :]
                    I_reference_rgb = dutils.tensor_lab2rgb(torch.cat((dutils.uncenter_l(I_reference_l), I_reference_ab), dim = 1))
                    features_B = ssnet.corr.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess = True)

                mask_warp_list = [mask_minus3_to_current, mask_minus2_to_current, mask_minus1_to_current, mask_add1_to_current, mask_add2_to_current, mask_add3_to_current]
                cpnet_warp_list = [cpnet_t_minus3_warp, cpnet_t_minus2_warp, cpnet_t_minus1_warp, cpnet_t_add1_warp, cpnet_t_add2_warp, cpnet_t_add3_warp]

                # SSNet forward propagation
                if use_long_connection:
                    ssnet_t, ssnet_sr_t, residual = ssnet(input_transform_frames[center_id], IB_lab, features_B, cpnet_frames[center_id], ssnet_t_minus1_warp, cpnet_warp_list, mask_warp_list)
                else:
                    ssnet_t, ssnet_sr_t, residual = ssnet(input_transform_frames[center_id], cpnet_frames[center_id], ssnet_t_minus1_warp, cpnet_warp_list, mask_warp_list)
                
            # Save
            if i == 0:
                bgr_minus3 = save_img(tensor_L = img_original_l[center_id - 3], tensor_ab = cpnet_frames[center_id - 3], save_name = imgroot[index][i + center_id - 3], opt = opt)
                bgr_minus2 = save_img(tensor_L = img_original_l[center_id - 2], tensor_ab = cpnet_frames[center_id - 2], save_name = imgroot[index][i + center_id - 2], opt = opt)
                bgr_minus1 = save_img(tensor_L = img_original_l[center_id - 1], tensor_ab = cpnet_frames[center_id - 1], save_name = imgroot[index][i + center_id - 1], opt = opt)
            if i == N - opt.iter_frames:
                bgr_add1 = save_img(tensor_L = img_original_l[center_id + 1], tensor_ab = cpnet_frames[center_id + 1], save_name = imgroot[index][i + center_id + 1], opt = opt)
                bgr_add2 = save_img(tensor_L = img_original_l[center_id + 2], tensor_ab = cpnet_frames[center_id + 2], save_name = imgroot[index][i + center_id + 2], opt = opt)
                bgr_add3 = save_img(tensor_L = img_original_l[center_id + 3], tensor_ab = cpnet_frames[center_id + 3], save_name = imgroot[index][i + center_id + 3], opt = opt)
            bgr = save_img(tensor_L = img_original_l[center_id], tensor_ab = ssnet_sr_t, save_name = imgroot[index][i + center_id], opt = opt)
            
            # Compute metrics
            # i == 0, the initial time step
            if i == 0:
                this_PSNR, this_SSIM = compute_metrics(bgr_minus3, gt_bgr_frames[center_id - 3])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id - 3, this_PSNR, this_SSIM))
                this_PSNR, this_SSIM = compute_metrics(bgr_minus2, gt_bgr_frames[center_id - 2])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id - 2, this_PSNR, this_SSIM))
                this_PSNR, this_SSIM = compute_metrics(bgr_minus1, gt_bgr_frames[center_id - 1])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id - 1, this_PSNR, this_SSIM))
                count = count + 3
            # 0 < i < N - opt.iter_frames
            this_PSNR, this_SSIM = compute_metrics(bgr, gt_bgr_frames[center_id])
            val_PSNR += this_PSNR
            val_SSIM += this_SSIM
            count = count + 1
            #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id, this_PSNR, this_SSIM))
            # i == N - opt.iter_frames, the last time step
            if i == N - opt.iter_frames:
                this_PSNR, this_SSIM = compute_metrics(bgr_add1, gt_bgr_frames[center_id + 1])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id + 1, this_PSNR, this_SSIM))
                this_PSNR, this_SSIM = compute_metrics(bgr_add2, gt_bgr_frames[center_id + 2])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id + 2, this_PSNR, this_SSIM))
                this_PSNR, this_SSIM = compute_metrics(bgr_add3, gt_bgr_frames[center_id + 3])
                val_PSNR += this_PSNR
                val_SSIM += this_SSIM
                #print('The %d-th video (%s), %d-th frame: PSNR: %.5f, SSIM: %.5f' % (index, imgroot[index][i].split('/')[0], i + center_id + 3, this_PSNR, this_SSIM))
                count = count + 3

    val_PSNR = val_PSNR / count
    val_SSIM = val_SSIM / count
    print('The average of %s, %s: PSNR: %.5f, average SSIM: %.5f' % (opt.save_rgb_path, opt.tag, val_PSNR, val_SSIM))
    # The average of ./result/rgb, DAVIS: PSNR: 25.95971, average SSIM: 0.95730
