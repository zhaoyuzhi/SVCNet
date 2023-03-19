import argparse
import os
import numpy as np
import cv2
from PIL import Image

# multi-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_files(path, keyword):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if keyword in filespath:
                ret.append(os.path.join(root, filespath.split(keyword)[0]))
    return ret

def mapping_func1(x):
    return 255 - x

def mapping_func2(x):
    return np.abs(128 - x)

def mapping_func3(x):
    return x // 2

def mapping_func4(x):
    return np.abs(np.power(x, 1.2) - x)

def mapping(img):
    # define parameters
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble_func1 = np.zeros((height, width, channels), np.uint8)
    scribble_func2 = np.zeros((height, width, channels), np.uint8)
    scribble_func3 = np.zeros((height, width, channels), np.uint8)
    scribble_func4 = np.zeros((height, width, channels), np.uint8)
    # perform mapping function
    for i in range(height):
        for j in range(width):
            if img[i, j, 0] > 0 or img[i, j, 1] > 0 or img[i, j, 2] > 0:
                scribble_func1[i, j, :] = mapping_func1(img[i, j, :])
                scribble_func2[i, j, :] = mapping_func2(img[i, j, :])
                scribble_func3[i, j, :] = mapping_func3(img[i, j, :])
                scribble_func4[i, j, :] = mapping_func4(img[i, j, :])
    scribble_func1 = np.clip(scribble_func1, 0, 255).astype(np.uint8)
    scribble_func2 = np.clip(scribble_func2, 0, 255).astype(np.uint8)
    scribble_func3 = np.clip(scribble_func3, 0, 255).astype(np.uint8)
    scribble_func4 = np.clip(scribble_func4, 0, 255).astype(np.uint8)
    return scribble_func1, scribble_func2, scribble_func3, scribble_func4

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseroot', type = str, \
        default = "F:\\submitted papers\\SVCNet\\code_evaluation\\fixed_color_scribbles\\color_point40_color_width5_256p", \
            help = 'baseroot')
    parser.add_argument('--saveroot', type = str, \
        default = "F:\\submitted papers\\SVCNet\\code_evaluation\\fixed_color_scribbles\\color_point40_color_width5_256p_mapping", \
            help = 'baseroot')
    parser.add_argument('--keyword', type = str, \
        default = "_256p.png", \
            help = 'keyword')
    parser.add_argument('--show', type = bool, default = True, help = 'show image color scribbles')
    parser.add_argument('--color_width', type = int, default = 10, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 5, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)

    ret = get_files(opt.baseroot, opt.keyword)
    
    for i in range(len(ret)):
        print(i, len(ret), ret[i])
        readpath = ret[i] + opt.keyword
        img = cv2.imread(readpath)
        scribble_func1, scribble_func2, scribble_func3, scribble_func4 = mapping(img)

        savefolder = os.path.join(opt.saveroot, 'diverse1', ret[i].split('\\')[-3], ret[i].split('\\')[-2])
        check_path(savefolder)
        savepath = os.path.join(savefolder, ret[i].split('\\')[-1] + opt.keyword)
        cv2.imwrite(savepath, scribble_func1)

        savefolder = os.path.join(opt.saveroot, 'diverse2', ret[i].split('\\')[-3], ret[i].split('\\')[-2])
        check_path(savefolder)
        savepath = os.path.join(savefolder, ret[i].split('\\')[-1] + opt.keyword)
        cv2.imwrite(savepath, scribble_func2)

        savefolder = os.path.join(opt.saveroot, 'diverse3', ret[i].split('\\')[-3], ret[i].split('\\')[-2])
        check_path(savefolder)
        savepath = os.path.join(savefolder, ret[i].split('\\')[-1] + opt.keyword)
        cv2.imwrite(savepath, scribble_func3)

        savefolder = os.path.join(opt.saveroot, 'diverse4', ret[i].split('\\')[-3], ret[i].split('\\')[-2])
        check_path(savefolder)
        savepath = os.path.join(savefolder, ret[i].split('\\')[-1] + opt.keyword)
        cv2.imwrite(savepath, scribble_func4)
