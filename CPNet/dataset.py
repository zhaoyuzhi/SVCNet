import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

class ScribbleColorizationDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_files(opt.base_root)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgpath = self.imglist[index]                                       # name of one image
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)                                                 # read one image (RGB)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        # convert to CIE Lab color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_l = lab[:, :, [0]]
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        
        # color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, \
            color_width = self.opt.color_width, color_point_prob = self.opt.color_point_prob)
        lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
        color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # saliency map
        #segpath = os.path.join(self.opt.seg_root, imgname)                   # path of one image
        segpath = imgpath.replace(self.opt.base_root, self.opt.seg_root)
        seg = cv2.imread(segpath, flags = cv2.IMREAD_GRAYSCALE)
        # image resize
        seg = cv2.resize(seg, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)

        # normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        seg = torch.from_numpy(seg.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        
        # img_l: 1 * 256 * 256; img_ab: 2 * 256 * 256; color_scribble: 2 * 256 * 256; sal: 1 * 256 * 256
        return img_l, img_ab, color_scribble_ab, seg

    def color_scribble(self, img, color_point, color_width, color_point_prob):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)
        if np.random.rand() > color_point_prob:
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
    
    def blurish(self, img, color_blur_width):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img

class ScribbleColorizationValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.text_readlines(opt.txt_root)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.base_root, imgname)                 # path of one image
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)                                                 # read one image (RGB)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        # convert to CIE Lab color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_l = lab[:, :, [0]]
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        # GT bgr
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, color_width = self.opt.color_width)
        #cv2.imshow('show', color_scribble)
        #cv2.waitKey(0)
        lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
        color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_bgr = torch.from_numpy(img_bgr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        
        # img_l: 1 * 256 * 256; img_ab: 2 * 256 * 256; color_scribble: 2 * 256 * 256
        return img_l, img_ab, img_bgr, color_scribble_ab, color_scribble, imgname

    def color_scribble(self, img, color_point, color_width):
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
    
    def blurish(self, img, color_blur_width):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img

class GivenScribbleColorizationValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.text_readlines(opt.txt_root)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.base_root, imgname)                 # path of one image
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)                                                 # read one image (RGB)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        # convert to CIE Lab color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_l = lab[:, :, [0]]
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        # GT bgr
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # color map
        scribblepath = os.path.join(self.opt.scribble_root, imgname.replace('.JPEG', '.png'))
        color_scribble = Image.open(scribblepath).convert('RGB')
        color_scribble = np.array(color_scribble)
        # image resize
        color_scribble = cv2.resize(color_scribble, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
        color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_bgr = torch.from_numpy(img_bgr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        
        # img_l: 1 * 256 * 256; img_ab: 2 * 256 * 256; color_scribble: 2 * 256 * 256
        return img_l, img_ab, img_bgr, color_scribble_ab, color_scribble, imgname

if __name__ == "__main__":

    img = Image.open('example.JPEG').convert('RGB')
    img = np.array(img)                                                 # read one image (RGB)
    cv2.imshow('img_bgr', img)
    cv2.waitKey(0)

    img = cv2.imread('example.JPEG')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    l = lab[:, :, [0]]
    a = lab[:, :, [1]]
    b = lab[:, :, [2]]
    l3 = np.concatenate((l, l, l), axis = 2)
    a3 = np.concatenate((a, a, a), axis = 2)
    b3 = np.concatenate((b, b, b), axis = 2)

    show = np.concatenate((img, lab, l3, a3, b3), axis = 1)

    cv2.imshow('show', show)
    cv2.waitKey(0)
