import os
import numpy as np
import cv2
import skimage.measure
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision as tv

import network
from network_vgg import PerceptualNet, VGGFeaNet

def create_generator(opt):
    # Initialize the networks
    generator = getattr(network, opt.pre_train_cpnet_type)(opt)
    print('Generator is created!')
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        generator = load_dict(generator, pretrained_net)
        print('Load the generator with %s' % opt.finetune_path)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_vggnet(vgg16_model_path):
    # Get the first 15 layers of vgg16, which is conv3_3
    vggnet = VGGFeaNet()
    # Pre-trained VGG-16
    vgg16 = torch.load(vgg16_model_path)
    load_dict(vggnet, vgg16)
    return vggnet

def create_perceptualnet(vgg16_model_path):
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = PerceptualNet(vgg16_model_path)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
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
def sample(input_img, gt_img, color_scribble, out_img, save_folder, epoch):
    # to cpu
    input_img = input_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    input_img = np.concatenate((input_img, input_img, input_img), axis = 2)                     # 256 * 256 * 3 (√)
    input_img = (input_img * 255).astype(np.uint8)
    gt_img = gt_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                           # 256 * 256 * 2
    gt_img = (gt_img * 255).astype(np.uint8)
    gt_img = np.concatenate((input_img[:, :, [0]], gt_img), axis = 2)                           # 256 * 256 * 3
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_Lab2BGR)                                            # 256 * 256 * 3 (√)
    color_scribble = color_scribble[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)           # 256 * 256 * 2 (√)
    color_scribble = (color_scribble * 255).astype(np.uint8)
    color_scribble = np.concatenate((input_img[:, :, [0]], color_scribble), axis = 2)           # 256 * 256 * 3
    color_scribble = cv2.cvtColor(color_scribble, cv2.COLOR_Lab2BGR)                            # 256 * 256 * 3 (√)
    out_img = out_img[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                         # 256 * 256 * 2
    out_img = (out_img * 255).astype(np.uint8)
    out_img = np.concatenate((input_img[:, :, [0]], out_img), axis = 2)                         # 256 * 256 * 3
    out_img = cv2.cvtColor(out_img, cv2.COLOR_Lab2BGR)                                          # 256 * 256 * 3 (√)
    # save
    img = np.concatenate((input_img, gt_img, color_scribble, out_img), axis = 1)
    imgname = os.path.join(save_folder, str(epoch) + '.png')
    cv2.imwrite(imgname, img)

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
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

if __name__ == "__main__":

    ret = text_readlines("./txt/ctest10k.txt")
    print(len(ret))
    