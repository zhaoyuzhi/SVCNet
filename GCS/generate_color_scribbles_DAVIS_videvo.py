import argparse
import os
import numpy as np
import cv2
from PIL import Image
import torch

import pwcnet

def create_pwcnet(opt):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(opt.pwcnet_path)
    if 'state_dict' in data.keys():
        flownet.load_state_dict(data['state_dict'])
    else:
        flownet.load_state_dict(data)
    print('PWCNet is loaded!')
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def color_scribble(img, color_point, color_width):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    times = color_point
    for i in range(times):
        # random selection
        rand_h = np.random.randint(color_width, height - color_width)
        rand_w = np.random.randint(color_width, width - color_width)
        # attach color points
        scribble[rand_h, rand_w, :] = img[rand_h, rand_w, :]
    return scribble

def widen_scribble(img, color_width):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i, j, 0] > 0 or img[i, j, 1] > 0 or img[i, j, 2] > 0:
                # define min and max
                min_h = i - (color_width - 1) // 2
                max_h = i + (color_width - 1) // 2
                min_w = j - (color_width - 1) // 2
                max_w = j + (color_width - 1) // 2
                # attach color points
                scribble[min_h:max_h, min_w:max_w, :] = img[i, j, :]
    return scribble

def blurish(img, color_blur_width):
    img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
    return img

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type = str, \
        default = 'DAVIS', \
            help = 'DAVIS | videvo')
    parser.add_argument('--pwcnet_path', type = str, \
        default = 'pwcNet-default.pytorch', \
            help = 'the load name of PWCNet')
    parser.add_argument('--txt_path', type = str, \
        default = './txt', \
            help = 'the path that contains class.txt')
    parser.add_argument('--baseroot', type = str, \
        default = './data/DAVIS_Videvo/val', \
            help = 'baseroot')
    parser.add_argument('--saveroot', type = str, \
        default = "./color_point40_color_width5_256p", \
            help = 'saveroot')
    parser.add_argument('--show', type = bool, default = True, help = 'show image color scribbles')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size') # 256, 128, 64
    parser.add_argument('--crop_size_w', type = int, default = 448, help = 'single patch size') # 448, 224, 128
    parser.add_argument('--color_point', type = int, default = 40, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble') # 5, 3
    parser.add_argument('--color_blur_width', type = int, default = 11, help = 'Gaussian blur width of each color scribble') # 3
    opt = parser.parse_args()
    print(opt)

    # PWCNet
    flownet = create_pwcnet(opt).cuda()

    # Inference for color scribbles
    imglist = text_readlines(os.path.join(opt.txt_path, opt.tag + '_test_imagelist.txt'))
    classlist = text_readlines(os.path.join(opt.txt_path, opt.tag + '_test_class.txt'))
    imgroot = [list() for i in range(len(classlist))]

    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(imglist):
            if imgname.split('/')[-2] == classname:
                imgroot[i].append(imgname)

    print('There are %d videos in the test set.' % (len(imgroot)))

    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):
        #index = 3
        N = len(imgroot[index])
        for i in range(N):
            imgpath = os.path.join(opt.baseroot, opt.tag, imgroot[index][i].split('/')[0], imgroot[index][i].split('/')[1])
            img = Image.open(imgpath).convert('RGB')
            img = np.array(img)
            h, w = img.shape[0], img.shape[1]
            img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
            if i == 0:
                s = color_scribble(img = img, color_point = opt.color_point, color_width = 1)
                s_previous = s.astype(np.float32) / 255.0
            if i > 0:
                img_previous_tensor = torch.from_numpy(img_previous.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
                s_tensor = torch.from_numpy(s_previous).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
                with torch.no_grad():
                    flow = pwcnet.PWCEstimate(flownet, img_previous_tensor, img_tensor, drange = True, reshape = True)
                    s_warp = pwcnet.PWCNetBackward(s_tensor, flow)
                s_previous = s_warp[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)
                s = (s_previous * 255).astype(np.uint8)

            print(h, w, s.sum())

            s_original_size = cv2.resize(s, (w, h), interpolation = cv2.INTER_NEAREST)
            widen_s_original_size = widen_scribble(s_original_size, opt.color_width)

            widen_s = widen_scribble(s, opt.color_width)
            img_previous = img.copy()

            '''
            if opt.show:
                assert s.sum() != widen_s.sum()
                matting = (0.7 * img + 0.3 * widen_s).astype(np.uint8)
                # img and s are in RGB format, then we need to convert them to BGR and show them
                show = np.concatenate((img, widen_s, matting), axis = 0)
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                cv2.imshow('test', show)
                cv2.waitKey(0)
            '''

            save_sub_folder_name = os.path.join(opt.saveroot, opt.tag, imgroot[index][i].split('/')[0])
            check_path(save_sub_folder_name)

            save_resized_scribble_name = os.path.join(save_sub_folder_name, imgroot[index][i].split('/')[1].split('.')[0] + '_' + str(opt.crop_size_h) + 'p.png')
            widen_s = cv2.cvtColor(widen_s, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_resized_scribble_name, widen_s)

            save_original_size_scribble_name = os.path.join(save_sub_folder_name, imgroot[index][i].split('/')[1].split('.')[0] + '.png')
            widen_s_original_size = cv2.cvtColor(widen_s_original_size, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_original_size_scribble_name, widen_s_original_size)
            
            save_resized_scribble_name = os.path.join(save_sub_folder_name, imgroot[index][i].split('/')[1].split('.')[0] + '_' + str(opt.crop_size_h) + 'p_not_widen.png')
            cv2.imwrite(save_resized_scribble_name, s)

            save_original_size_scribble_name = os.path.join(save_sub_folder_name, imgroot[index][i].split('/')[1].split('.')[0] + '_not_widen.png')
            cv2.imwrite(save_original_size_scribble_name, s_original_size)
            