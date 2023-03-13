import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import networks.cpnet as cpnet_lib
import networks.ssnet as ssnet_lib
import networks.pwcnet as pwcnet
import networks.network_others as network_others
import networks.network_utils as network_utils

# ----------------------------------------
#                Networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def create_generator(opt, tag = 'train'):
    # Initialize the networks
    cpnet = getattr(cpnet_lib, opt.pre_train_cpnet_type)(opt)
    ssnet = getattr(ssnet_lib, opt.pre_train_ssnet_type)(opt)
    weights_init(cpnet, init_type = 'normal', init_gain = 0.02)
    weights_init(ssnet, init_type = 'normal', init_gain = 0.02)
    print('Generators are created!')
    # Init the networks
    if tag == 'train':
        # for CPNet, normally we will give a load path for initialization, but this is for ablation study
        if opt.cpnet_path:
            pretrained_net = torch.load(opt.cpnet_path)
            cpnet = load_dict(cpnet, pretrained_net)
        # for SSNet at training, normally we will not give a load path, but this is only for continuing training
        if opt.ssnet_path:
            pretrained_net = torch.load(opt.ssnet_path)
            ssnet = load_dict(ssnet, pretrained_net)
        else:
            # for subnets of SSNet, normally we will give a load path for initialization, but this is for ablation study
            if opt.warn_path:
                pretrained_net = torch.load(opt.warn_path)
                ssnet.warn = load_dict(ssnet.warn, pretrained_net)
            if opt.corrnet_vgg_path:
                pretrained_net = torch.load(opt.corrnet_vgg_path)
                ssnet.corr.vggnet = load_dict(ssnet.corr.vggnet, pretrained_net)
            if opt.corrnet_nonlocal_path:
                pretrained_net = torch.load(opt.corrnet_nonlocal_path)
                ssnet.corr.nonlocal_net = load_dict(ssnet.corr.nonlocal_net, pretrained_net)
            if opt.srnet_path:
                pretrained_net = torch.load(opt.srnet_path)
                ssnet.srnet = load_dict(ssnet.srnet, pretrained_net)
    elif tag == 'val':
        pretrained_net = torch.load(opt.cpnet_path)
        cpnet = load_dict(cpnet, pretrained_net)
        pretrained_net = torch.load(opt.ssnet_path)
        ssnet = load_dict(ssnet, pretrained_net)
    print('Load the CPNet and SSNet for %s stage' % tag)
    return cpnet, ssnet

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network_others.PatchDiscriminator(opt)
    print('Discriminator is created!')
    return discriminator

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

def create_vggnet(opt):
    # Get the first 15 layers of vgg16, which is conv3_3
    vggnet = network_utils.VGGFeaNet()
    # Pre-trained VGG-16
    vgg16 = torch.load(opt.perceptual_path)
    load_dict(vggnet, vgg16)
    return vggnet

def create_perceptualnet(opt):
    # Pre-trained VGG-16
    vgg16 = torch.load(opt.perceptual_path)
    # Get the first 16 layers of vgg16, which is conv3_3
    perceptualnet = network_others.PerceptualNet()
    # Update the parameters
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#                Datasets
# ----------------------------------------
class SubsetSeSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return len(self.indices)

def create_dataloader(dataset, opt):
    #dataloader = DataLoader(dataset = dataset, batch_size = opt.batch_size, num_workers = opt.num_workers, shuffle = True, pin_memory = True)
    # Generate random index
    indices = np.random.permutation(len(dataset))
    indices = np.tile(indices, opt.batch_size)
    # Generate data sampler and loader
    datasampler = SubsetSeSampler(indices)
    dataloader = DataLoader(dataset = dataset, batch_size = opt.batch_size, num_workers = opt.num_workers, sampler = datasampler, pin_memory = True)
    return dataloader

# ----------------------------------------
#             Path processing
# ----------------------------------------
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

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

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

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def sample(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, save_format = 'png'):
    # sample_folder: the path that saves all the sample images
    # sample_name: the specific iteration of this group of images
    # img_list / name_list: lists that save all the files and names to be saved

    # Save image one-by-one
    for i in range(len(img_list)):

        img = img_list[i]

        # Process img_copy and do not destroy the data of img
        if i == 0 and img.shape[1] == 1:
            img = torch.cat((img, img, img), 1)
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = (img_copy * pixel_max_cnt)
        img_copy = np.clip(img_copy, 0, pixel_max_cnt).astype(np.uint8)

        # The first one should be grayscale
        if i == 0:
            grayscale = img_copy.copy()
        else:
            img_copy = np.concatenate((grayscale[:, :, [0]], img_copy), axis = 2)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_Lab2BGR)
        
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.' + save_format
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def sample_nlimg(sample_folder, sample_name, img, name, save_format = 'png'):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    save_img_name = sample_name + '_' + name + '.' + save_format
    save_img_path = os.path.join(sample_folder, save_img_name)
    cv2.imwrite(save_img_path, img)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#                  Others
# ----------------------------------------
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def scribble_sampling(img, color_point, color_width):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    for i in range(color_point):
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

def scribble_sampling_uniform(img, color_point, color_width):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)
    # pre-define heights and widths
    heights = []
    widths = []
    for i in range(color_point):
        this_height = height // (color_point + 1) * (i + 1)
        heights.append(this_height)
        this_width = width // (color_point + 1) * (i + 1)
        widths.append(this_width)
    for j in range(color_point):
        for k in range(color_point):
            # random selection
            rand_h = heights[j]
            rand_w = widths[k]
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
