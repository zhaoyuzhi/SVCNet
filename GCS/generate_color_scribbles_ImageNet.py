import argparse
import os
import numpy as np
import cv2
from PIL import Image

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

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

'''
# Color map
color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, color_width = self.opt.color_width)
lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
#color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
'''

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type = str, \
        default = 'videvo', \
            help = 'DAVIS | videvo')
    parser.add_argument('--txt_path', type = str, \
        default = './txt', \
            help = 'the path that contains class.txt')
    parser.add_argument('--baseroot', type = str, \
        default = '/home/zyz/Documents/dataset/ILSVRC2012_val_256', \
            help = 'baseroot')
    parser.add_argument('--saveroot', type = str, \
        default = "./color_point40_color_width5/ImageNet", \
            help = 'saveroot')
    parser.add_argument('--show', type = bool, default = True, help = 'show image color scribbles')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--color_point', type = int, default = 40, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 5, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)

    imglist = get_files(opt.baseroot)

    check_path(opt.saveroot)

    N = len(imglist)
    for i in range(N):
        imgpath = imglist[i]
        imgname = imgpath.split('/')[-1]
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)
        h, w = img.shape[0], img.shape[1]
        img = cv2.resize(img, (opt.crop_size_w, opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        s = color_scribble(img = img, color_point = opt.color_point, color_width = 1)
        widen_s = widen_scribble(s, opt.color_width)

        print(h, w, s.sum(), i, N)
        
        save_resized_scribble_name = os.path.join(opt.saveroot, imgname.split('.')[0] + '.png')
        widen_s = cv2.cvtColor(widen_s, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_resized_scribble_name, widen_s)
