import os
import math
import random
import torch
import numpy as np
import cv2
import torchvision.transforms as transform_lib
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from skimage.draw import random_shapes
from skimage.filters import gaussian
from skimage.transform import resize

l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0

# ----------------------------------------
#    data pre-processing for the CorrNet
# ----------------------------------------
def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

class RGB2Lab(object):
    def __init__(self):
        pass
    def __call__(self, inputs):
        return color.rgb2lab(inputs)

def to_mytensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    pic_arr = np.array(pic)
    if pic_arr.ndim == 2:
        pic_arr = pic_arr[..., np.newaxis]
    img = torch.from_numpy(pic_arr.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        return img.float()  # no normalize .div(255)
    else:
        return img

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, inputs):
        return to_mytensor(inputs)

def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError("tensor is not a torch image.")
    # TODO: make efficient
    if tensor.size(0) == 1:
        tensor.sub_(mean).div_(std)
    else:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return tensor

class Normalize(object):
    def __init__(self):
        pass
    def __call__(self, inputs):
        inputs[0:1, :, :] = normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs

def create_transform(opt):
    transform = transform_lib.Compose(
        [transform_lib.Resize([opt.crop_size_h, opt.crop_size_w]), RGB2Lab(), ToTensor(), Normalize()]
    )
    return transform

# ----------------------------------------
#    data pre-processing for the CorrNet
# ----------------------------------------
# CPNet related
def cpnet_ab_to_cv2_rgb(grayscale, ab):
    B, _, H, W = grayscale.shape
    rgb = np.zeros([B, 3, H, W], dtype = np.uint8)
    for i in range(B):
        grayscale_i = grayscale[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)               # 256 * 256 * 1
        ab_i = ab[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                             # 256 * 256 * 2
        lab_i = np.concatenate((grayscale_i, ab_i), axis = 2)                                   # 256 * 256 * 3
        lab_i = np.clip(lab_i, 0, 1)
        lab_i = (lab_i * 255).astype(np.uint8)
        lab_i = cv2.cvtColor(lab_i, cv2.COLOR_Lab2RGB)                                          # 256 * 256 * 3 (√)
        lab_i = lab_i.transpose(2, 0, 1)                                                        # 3 * 256 * 256
        rgb[i, :, :, :] = lab_i
    return rgb

def cpnet_ab_to_PIL_rgb(grayscale, ab):
    B, _, H, W = grayscale.shape
    rgb = []
    for i in range(B):
        grayscale_i = grayscale[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)               # 256 * 256 * 1
        ab_i = ab[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                             # 256 * 256 * 2
        lab_i = np.concatenate((grayscale_i, ab_i), axis = 2)                                   # 256 * 256 * 3
        lab_i = np.clip(lab_i, 0, 1)
        lab_i = (lab_i * 255).astype(np.uint8)
        lab_i = cv2.cvtColor(lab_i, cv2.COLOR_Lab2RGB)                                          # 256 * 256 * 3 (√)
        lab_i = Image.fromarray(lab_i)
        rgb.append(lab_i)
    return rgb

def cpnet_ab_to_tensor_rgb(grayscale, ab):
    B, _, H, W = grayscale.shape
    rgb = torch.zeros([B, 3, H, W], dtype = torch.float)
    for i in range(B):
        grayscale_i = grayscale[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)               # 256 * 256 * 1
        ab_i = ab[i, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                             # 256 * 256 * 2
        lab_i = np.concatenate((grayscale_i, ab_i), axis = 2)                                   # 256 * 256 * 3
        lab_i = np.clip(lab_i, 0, 1)
        lab_i = (lab_i * 255).astype(np.uint8)
        lab_i = cv2.cvtColor(lab_i, cv2.COLOR_Lab2RGB)                                          # 256 * 256 * 3 (√)
        lab_i = torch.from_numpy(lab_i.astype(np.float32) / 255.0).permute(2, 0, 1)             # 3 * 256 * 256
        rgb[i, :, :, :] = lab_i
    rgb = rgb.contiguous().cuda()
    return rgb

# CorrNet related
def uncenter_l(l):
    return l * l_norm + l_mean

xyz_from_rgb = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
)

def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb


# ----------------------------------------
#    create scribbles for numpy ndarray
# ----------------------------------------
def color_scribble(img, color_point, color_width, use_color_point):
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

def scribble_sampling(img, color_point):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    for i in range(color_point):
        # random selection
        rand_h = np.random.randint(height)
        rand_w = np.random.randint(width)
        # attach color points
        scribble[rand_h, rand_w, :] = img[rand_h, rand_w, :]
    return scribble

def scribble_expand(img, color_width):
    # define subfunc
    def expand_one(img):
        # define output
        img_expand = np.zeros_like(img)
        for (H_shift, W_shift) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            img_expand = np.maximum(img_expand, np.roll(img, (H_shift, W_shift), (0, 1)))
        return img_expand
    # suppose the input image is a scribble image, where each scribble only contains a single pixel
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    half_len = (color_width - 1) // 2
    for i in range(half_len):
        img = expand_one(img)
    img[:half_len, :, :] = 0
    img[(height - half_len):height, :, :] = 0
    img[:, :half_len, :] = 0
    img[:, (width - half_len):width, :] = 0
    return img

# ----------------------------------------
#    create scribbles for PyTorch tensor
# ----------------------------------------
def scribble_sampling_tensor(img, color_point):
    height = img.shape[2]
    width = img.shape[3]
    scribble = torch.zeros_like(img).to(img.device)
    for i in range(color_point):
        # random selection
        rand_h = np.random.randint(height)
        rand_w = np.random.randint(width)
        # attach color points
        scribble[:, :, rand_h, rand_w] = img[:, :, rand_h, rand_w]
    return scribble

def scribble_expand_tensor(img, color_width):
    # define subfunc
    def expand_one(img):
        # define output
        img_expand = torch.zeros_like(img)
        for (H_shift, W_shift) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            img_expand = torch.maximum(img_expand, torch.roll(img, (H_shift, W_shift), (2, 3)))
        return img_expand
    # suppose the input image is a scribble image, where each scribble only contains a single pixel
    height = img.shape[2]
    width = img.shape[3]
    half_len = (color_width - 1) // 2
    for i in range(half_len):
        img = expand_one(img)
    img[:, :, :half_len, :] = 0
    img[:, :, (height - half_len):height, :] = 0
    img[:, :, :, :half_len] = 0
    img[:, :, :, (width - half_len):width] = 0
    return img

def tensor_to_numpy(img):
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :].astype(np.uint8)
    return img

if __name__ == '__main__':

    imgpath = '/home/zyz/Documents/svcnet/2dataset_RGB/DAVIS/cows/00040.jpg'
    img = cv2.imread(imgpath)                                       # H, W, 3
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)     # 1, 3, H, W

    '''
    scribble = scribble_sampling(img, 40)
    cv2.imwrite('show.png', scribble)
    scribble_expand = scribble_expand(scribble, 5)
    cv2.imwrite('show2.png', scribble_expand)
    '''

    scribble = scribble_sampling_tensor(img_t, 40)
    cv2.imwrite('show.png', tensor_to_numpy(scribble))
    scribble_expand = scribble_expand_tensor(scribble, 5)
    cv2.imwrite('show2.png', tensor_to_numpy(scribble_expand))
