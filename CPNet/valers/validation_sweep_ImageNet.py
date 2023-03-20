import argparse
import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils

def color_scribble(img, color_point, color_width):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    if color_point > 0:
        times = np.random.randint(color_point)
        for i in range(times):
            # random selection
            rand_h = np.random.randint(height)
            rand_w = np.random.randint(width)
            # define min and max
            min_h = rand_h - (color_width - 1) // 2
            max_h = rand_h + (color_width - 1) // 2
            min_w = rand_w - (color_width - 1) // 2
            max_w = rand_w + (color_width - 1) // 2
            min_h = max(min_h, 0)
            min_w = max(min_w, 0)
            max_h = min(max_h, height)
            max_w = min(max_w, width)
            # attach color points
            scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]
    return scribble
    
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
    parser.add_argument('--txt_root', type = str, default = "./txt/ctest10k.txt", help = 'the base training folder')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'size of image')
    # color scribble parameters
    parser.add_argument('--color_point', type = int, default = 40, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 11, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)
    
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
    imgroot = utils.text_readlines(opt.txt_root)
    
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    # Define the scribble and metric list
    scrib_list = np.arange(0, opt.color_point + 1)
    val_PSNR_list = np.zeros_like(scrib_list)
    val_SSIM_list = np.zeros_like(scrib_list)
    
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):

        # Read images
        imgname = imgroot[index]                                        # name of one image
        imgpath = os.path.join(opt.base_root, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gt_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_bgr = torch.from_numpy(gt_bgr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()

        img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_l = lab[:, :, [0]]
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
        
        # forward propagation
        with torch.no_grad():
            for j in range(opt.color_point + 1):
                
                scribble = color_scribble(img = img, color_point = j, color_width = opt.color_width)
                scribble = cv2.cvtColor(scribble, cv2.COLOR_RGB2Lab)
                scribble = np.concatenate((scribble[:, :, [1]], scribble[:, :, [2]]), axis = 2)
                #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
                scribble = torch.from_numpy(scribble.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()

                out = generator(img_l, scribble)
                
                if isinstance(out, tuple):
                    img_out = out[0]
                    seg_out = out[1]
                else:
                    img_out = out
                
                # PSNR
                assert img_out.shape == img_ab.shape
                this_PSNR = utils.psnr(img_out, img_ab, 1) * img_ab.shape[0]
                val_PSNR_list[j] = val_PSNR_list[j] + this_PSNR
                this_SSIM = utils.ssim(img_out, img_ab) * img_ab.shape[0]
                val_SSIM_list[j] = val_SSIM_list[j] + this_SSIM
                #print(j, this_PSNR, this_SSIM)
        
        print('The %d-th image' % (index))

    val_PSNR_list = val_PSNR_list / len(imgroot)
    val_SSIM_list = val_SSIM_list / len(imgroot)
    for k in range(opt.color_point + 1):
        print('%s: Num of scrib: %d, PSNR: %.5f, SSIM: %.5f' % (opt.pre_train_cpnet_type, k, val_PSNR_list[k], val_SSIM_list[k]))
