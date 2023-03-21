import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

def convert_lab_to_bgr(input_img, out_img):
    input_img = input_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)             # 256 * 256 * 1
    out_img = out_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                 # 256 * 256 * 2
    out_img = np.concatenate((input_img, out_img), axis = 2)                            # 256 * 256 * 3
    out_img = (out_img * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_Lab2BGR)                                  # 256 * 256 * 3 (√)
    return out_img

def recover_tensor_to_ndarray(img):
    img = img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                         # 256 * 256 * 3
    img = (img * 255).astype(np.uint8)
    return img

def recover_ndarray_to_tensor(img):
    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return img

def convert_seg(out_img):
    out_img = out_img[0, 0, :, :].data.cpu().numpy()                                    # 256 * 256 * 1
    out_img = (out_img * 255).astype(np.uint8)
    return out_img

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train_cpnet_type', type = str, default = 'CPNet_VGG16_Seg', help = 'pre_train_cpnet_type')
    parser.add_argument('--save_rgb_path', type = str, \
        default = './result_given_scribble_ImageNet', \
            help = 'save the generated rgb image to certain path')
    parser.add_argument('--finetune_path', type = str, \
        default = './trained_models/CPNet/models_1st/CPNet_VGG16_Seg/cpnet_epoch20_batchsize32.pth', \
            help = 'the load name of models')
    parser.add_argument('--vgg_name', type = str, default = "./trained_models/Others/vgg16_pretrained.pth", help = 'pre-trained vgg')
    # Training parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    # Dataset parameters
    parser.add_argument('--base_root', type = str, \
        default = "./data/ILSVRC2012/ILSVRC2012_val_256", \
            help = 'the base validation folder')
    parser.add_argument('--scribble_root', type = str, \
        default = './color_point40_color_width5_ImageNet', \
            help = 'the base validation folder')
    parser.add_argument('--txt_root', type = str, default = "./txt/ctest10k.txt", help = 'the base training folder')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    save_rgb_sub_folder_name = opt.save_rgb_path
    utils.check_path(save_rgb_sub_folder_name)
    
    # ----------------------------------------
    #       Initialize testing network
    # ----------------------------------------
    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    generator = generator.cuda()
    
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------
    # Define the dataset
    testset = dataset.GivenScribbleColorizationValDataset(opt)
    dataloader = DataLoader(testset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    # forward
    val_PSNR = 0
    val_SSIM = 0
    
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for batch_idx, (grayscale, gt_ab, gt_bgr, color_scribble, color_scribble_save, imgname) in enumerate(dataloader):

        # Load and put to cuda
        grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
        gt_bgr = gt_bgr.cuda()                                          # out: [B, 2, 256, 256]
        color_scribble = color_scribble.cuda()                          # out: [B, 2, 256, 256]
        imgname = imgname[0]

        # forward propagation
        with torch.no_grad():
            out = generator(grayscale, color_scribble)

            if isinstance(out, tuple):
                img_out = out[0]
                seg_out = out[1]
            else:
                img_out = out
            
        bgr = convert_lab_to_bgr(grayscale, img_out)
        save_rgb_name = os.path.join(save_rgb_sub_folder_name, imgname.replace('.JPEG', '.png'))
        cv2.imwrite(save_rgb_name, bgr)
        
        #seg = convert_seg(seg_out)
        #save_seg_name = os.path.join(save_rgb_sub_folder_name, imgname.split('.')[0] + '_seg.jpg')
        #cv2.imwrite(save_seg_name, seg)
        
        # PSNR
        bgr = recover_ndarray_to_tensor(bgr).cuda()
        assert bgr.shape == gt_bgr.shape
        this_PSNR = utils.psnr(bgr, gt_bgr, 1) * gt_bgr.shape[0]
        val_PSNR += this_PSNR
        this_SSIM = utils.ssim(bgr, gt_bgr) * gt_bgr.shape[0]
        val_SSIM += this_SSIM
        print('The %d-th image: Name: %s PSNR: %.5f, SSIM: %.5f' % (batch_idx + 1, imgname, this_PSNR, this_SSIM))

    val_PSNR = val_PSNR / len(testset)
    val_SSIM = val_SSIM / len(testset)
    print('The average of %s: PSNR: %.5f, SSIM: %.5f' % (save_rgb_sub_folder_name, val_PSNR, val_SSIM))
