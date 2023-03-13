import argparse
import os
import cv2
from PIL import Image
import numpy as np
import torch

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

def convert_lab_to_bgr(input_img, out_img):
    input_img = input_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)             # 256 * 256 * 1
    out_img = out_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                 # 256 * 256 * 2
    out_img = np.concatenate((input_img, out_img), axis = 2)                            # 256 * 256 * 3
    out_img = (out_img * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_Lab2BGR)                                  # 256 * 256 * 3 (√)
    return out_img

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--tag', type = str, \
        default = 'videvo', \
            help = 'DAVIS | videvo')
    parser.add_argument('--save_rgb_path', type = str, \
        default = './result', \
            help = 'save the generated rgb image to certain path')
    parser.add_argument('--finetune_path', type = str, \
        default = './models/cpnet_epoch40_batchsize32.pth', help = 'the load name of models')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--mid_channels', type = int, default = 128, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    # Dataset parameters
    parser.add_argument('--base_root', type = str, \
        default = 'F:\\dataset, my paper related\\VCGAN dataset\\test\\input', \
            help = 'the base training folder')
    parser.add_argument('--scribble_root', type = str, \
        default = "E:\\code\\Scribble-based Video Colorization\\experiment\\fixed_color_scribbles", \
            help = 'the base training folder')
    parser.add_argument('--txt_root', type = str, \
        default = "./txt", \
            help = 'the base training folder')
    parser.add_argument('--crop_size_h', type = int, default = 128, help = 'single patch size') # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'single patch size') # second stage (128p, 256p, 448p): 256, 448, 832
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
    imgroot = define_dataset(opt)
    
    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):
        N = len(imgroot[index])
        for i in range(N):
            # Read images
            img_path = os.path.join(opt.base_root, opt.tag, imgroot[index][i].split('/')[0], imgroot[index][i].split('/')[1])
            scribble_path = os.path.join(opt.scribble_root, opt.tag, imgroot[index][i].split('/')[0], imgroot[index][i].split('/')[1].split('.')[0] + '_128p.png')
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            #h, w = img.shape[0], img.shape[1]
            img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            img_l = lab[:, :, [0]]
            img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
            img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
            scribble = Image.open(scribble_path).convert('RGB')
            scribble = np.array(scribble)
            scribble = cv2.cvtColor(scribble, cv2.COLOR_RGB2Lab)
            scribble = np.concatenate((scribble[:, :, [1]], scribble[:, :, [2]]), axis = 2)
            scribble = torch.from_numpy(scribble.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
            
            # Inference
            with torch.no_grad():
                col = generator(img_l, scribble)
                
            # Save
            bgr = convert_lab_to_bgr(img_l, col)

            save_rgb_sub_folder_name = os.path.join(opt.save_rgb_path, opt.tag, imgroot[index][i].split('/')[0])
            utils.check_path(save_rgb_sub_folder_name)

            save_rgb_name = os.path.join(save_rgb_sub_folder_name, imgroot[index][i].split('/')[1].split('.')[0] + '_col_' + str(opt.crop_size_h) + 'p.png')
            cv2.imwrite(save_rgb_name, bgr)
            
            print('Video %d, Frame %d is saved' % (index, i))
