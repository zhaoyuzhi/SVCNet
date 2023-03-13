import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import lmdb

import datasets.data_utils as dutils

class MultiFramesDataset(Dataset):
    def __init__(self, opt, imglist, classlist):
        # If you want to use this code, please ensure:
        # 1. the structure of the training set should be given as: baseroot + classname (many classes) + imagename (many images)
        # 2. all the input images should be categorized, and each folder contains all the images of this class
        # 3. all the name of images should be given as: 00000.jpg, 00001.jpg, 00002.jpg, ... , 00NNN.jpg (if possible, not mandatary)
        # Input:
        # 1. opt: all the options
        # 2. imglist: all the "classname (many classes) + imagename (many images)"
        # 2. classlist: all the "classname (many classes)"
        # Note that:
        # 1. self.baseroot: the overall base
        # 2. self.classlist: the second base, it could not be used in get_item
        # 3. self.imgroot: the relative root for each image, and classified by categories (二维列表)
        # Inference:
        # number of classes = len(self.classlist) = len(self.imgroot)
        # number of images in a specific class = len(self.imgroot[k])
        # this dataset is fair for each class, not for each image, because the number of images in different classes is different
        self.opt = opt                                                  # baseroot is the base of all images
        self.imglist = imglist                                          # imglist should contain the category name of the series of frames + image names, in order
        self.classlist = classlist                                      # classlist should contain the category name of the series of frames
        self.imgroot = [list() for i in range(len(classlist))]          # imgroot contains the relative path of all images
        # Calculate the whole number of each class
        for i, classname in enumerate(self.classlist):
            for j, imgname in enumerate(imglist):
                if imgname.split('/')[-2] == classname:
                    self.imgroot[i].append(imgname)
        self.transform = dutils.create_transform(opt)

    def color_scribble(self, img, color_point, color_width, use_color_point):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)
        if use_color_point:
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

    def get_lab(self, imgpath, use_color_point):
        # Pre-processing, let all the images are in RGB color space
        img = Image.open(imgpath).convert('RGB')
        img = np.array(img)                                             # read one image (RGB)

        # Large scale RGB / Lab
        img_large = cv2.resize(img, (self.opt.crop_size_w * self.opt.upsample_ratio, self.opt.crop_size_h * self.opt.upsample_ratio), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img_large, cv2.COLOR_RGB2Lab)                # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img_large_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)

        # RGB / Lab
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)                      # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img_l = lab[:, :, [0]]
        img_l = np.concatenate((img_l, img_l, img_l), axis = 2)
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        
        # Color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, \
            color_width = self.opt.color_width, use_color_point = use_color_point)
        lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
        color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        
        # Normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_large_ab = torch.from_numpy(img_large_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        return img_l, img_ab, color_scribble_ab, img_large_ab

    def get_l(self, imgpath):
        img_l_transform = self.transform(Image.open(imgpath))[[0], :, :]
        return img_l_transform

    def get_seg(self, segpath):
        seg = cv2.imread(segpath, cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        seg = torch.from_numpy(seg.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return seg

    def __getitem__(self, index):
        # Choose a category of dataset, it is fair for each dataset to be chosen
        N = len(self.imgroot[index])
        # Pre-define the starting frame index in 0 ~ N - opt.iter_frames
        T = random.randint(0, N - self.opt.iter_frames - 1)
        # Sample from T to T + opt.iter_frames
        in_part = []
        in_transform_part = []
        out_part = []
        color_scribble_part = []
        out_large_part = []
        if np.random.rand() > self.opt.color_point_prob:
            use_color_point = True
        else:
            use_color_point = False
        for i in range(T, T + self.opt.iter_frames):
            # Define image path for an image
            imgpath = os.path.join(self.opt.baseroot, self.imgroot[index][i])
            # get_lab function
            img_l, img_ab, color_scribble_ab, img_large_ab = self.get_lab(imgpath, use_color_point)
            in_part.append(img_l)
            out_part.append(img_ab)
            color_scribble_part.append(color_scribble_ab)
            out_large_part.append(img_large_ab)
            # transform function
            img_l_transform = self.get_l(imgpath)
            in_transform_part.append(img_l_transform)
        # Each instance is paired
        return in_part, in_transform_part, out_part, color_scribble_part, out_large_part
    
    def __len__(self):
        return len(self.classlist)

class MultiFramesDataset_lmdb(Dataset):
    def __init__(self, opt, imglist, classlist):
        self.opt = opt                                                  # baseroot is the base of all images
        self.imglist = imglist                                          # imglist should contain the category name of the series of frames + image names, in order
        self.classlist = classlist                                      # classlist should contain the category name of the series of frames
        self.imgroot = [list() for i in range(len(classlist))]          # imgroot contains the relative path of all images
        # Calculate the whole number of each class
        for i, classname in enumerate(self.classlist):
            for j, imgname in enumerate(imglist):
                if imgname.split('/')[-2] == classname:
                    self.imgroot[i].append(imgname)
        self.env = lmdb.open(
                opt.baseroot,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        self.transform = dutils.create_transform(opt)

    def color_scribble(self, img, color_point, color_width, use_color_point):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)
        if use_color_point:
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

    def get_lab(self, imgpath, use_color_point):
        # Pre-processing, let all the images are in RGB color space
        with self.env.begin(write=False) as txn:
            key = imgpath.encode('utf-8')
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        img = np.array(img)                                             # read one image (RGB)

        # Large scale RGB / Lab
        img_large = cv2.resize(img, (self.opt.crop_size_w * self.opt.upsample_ratio, self.opt.crop_size_h * self.opt.upsample_ratio), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img_large, cv2.COLOR_RGB2Lab)                # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img_large_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)

        # RGB / Lab
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)                      # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img_l = lab[:, :, [0]]
        img_l = np.concatenate((img_l, img_l, img_l), axis = 2)
        img_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        
        # Color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, \
            color_width = self.opt.color_width, use_color_point = use_color_point)
        lab = cv2.cvtColor(color_scribble, cv2.COLOR_RGB2Lab)
        color_scribble_ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble_ab = torch.from_numpy(color_scribble_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        
        # Normalization
        img_l = torch.from_numpy(img_l.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_ab = torch.from_numpy(img_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_large_ab = torch.from_numpy(img_large_ab.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        return img_l, img_ab, color_scribble_ab, img_large_ab

    def get_l(self, imgpath):
        # Pre-processing, let all the images are in RGB color space
        with self.env.begin(write=False) as txn:
            key = imgpath.encode('utf-8')
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        img_l_transform = self.transform(img)[[0], :, :]
        return img_l_transform

    def get_seg(self, segpath):
        seg = cv2.imread(segpath, cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, (self.opt.crop_size_w, self.opt.crop_size_h), interpolation = cv2.INTER_CUBIC)
        seg = torch.from_numpy(seg.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return seg

    def __getitem__(self, index):
        # Choose a category of dataset, it is fair for each dataset to be chosen
        N = len(self.imgroot[index])
        # Pre-define the starting frame index in 0 ~ N - opt.iter_frames
        T = random.randint(0, N - self.opt.iter_frames - 1)
        # Sample from T to T + opt.iter_frames
        in_part = []
        in_transform_part = []
        out_part = []
        color_scribble_part = []
        out_large_part = []
        if np.random.rand() > self.opt.color_point_prob:
            use_color_point = True
        else:
            use_color_point = False
        for i in range(T, T + self.opt.iter_frames):
            # Define image path for an image
            imgpath = self.imgroot[index][i]
            # get_lab function
            img_l, img_ab, color_scribble_ab, img_large_ab = self.get_lab(imgpath, use_color_point)
            in_part.append(img_l)
            out_part.append(img_ab)
            color_scribble_part.append(color_scribble_ab)
            out_large_part.append(img_large_ab)
            # transform function
            img_l_transform = self.get_l(imgpath)
            in_transform_part.append(img_l_transform)
        # Each instance is paired
        return in_part, in_transform_part, out_part, color_scribble_part, out_large_part
    
    def __len__(self):
        return len(self.classlist)

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
