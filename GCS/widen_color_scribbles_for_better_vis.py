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

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseroot', type = str, \
        default = "./color_point40_color_width5_256p", \
            help = 'baseroot')
    parser.add_argument('--keyword', type = str, \
        default = "_256p.png", \
            help = 'keyword')
    parser.add_argument('--save_keyword', type = str, \
        default = "_vis.png", \
            help = 'save_keyword')
    parser.add_argument('--show', type = bool, default = True, help = 'show image color scribbles')
    parser.add_argument('--color_width', type = int, default = 10, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 5, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)

    ret = get_files(opt.baseroot, opt.keyword)

    for i in range(len(ret)):
        print(i, len(ret), ret[i])
        readpath = ret[i] + opt.keyword
        savepath = ret[i] + opt.save_keyword
        img = cv2.imread(readpath)
        img = widen_scribble(img, opt.color_width)
        cv2.imwrite(savepath, img)
    