import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from skimage.draw import random_shapes
from skimage.filters import gaussian
from skimage.transform import resize

l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0

###
def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow = 8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    if pred_lab.shape[0] == 1:
        grid_lab = pred_lab[0, ...].numpy().astype("float64")
    else:
        grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    grid_rgb =(np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8") # RGB format
    return grid_rgb

def corr_lab_to_cv2_rgb(IA_lab, IA_l_large):
    # upsampling
    I_current_ab_predict = IA_lab[:, 1:3, :, :]
    #curr_predict = F.interpolate(I_current_ab_predict.data.cpu(), scale_factor = 2, mode = "bilinear")
    curr_predict = I_current_ab_predict.data.cpu()
    IA_predict_rgb = batch_lab2rgb_transpose_mc(IA_l_large, curr_predict)
    return IA_predict_rgb

def cv2_rgb_to_tensor_ab(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    ab = np.concatenate((lab[:, :, [1]], lab[:, :, [2]]), axis = 2)
    ab = torch.from_numpy(ab.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    return ab

###
# denormalization for l
def uncenter_l(l):
    return l * l_norm + l_mean

def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l)
    l_uncenter = l_uncenter / (2 * l_mean)
    return torch.cat((l_uncenter, l_uncenter, l_uncenter), dim=1)

# ----------------------------------------
#            Backbone Network
# ----------------------------------------
class VGGFeaNet(nn.Module):
    def __init__(self):
        super(VGGFeaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),                      # 0   conv1_1
            nn.ReLU(inplace = True),                        # 1   relu1_1
            nn.Conv2d(64, 64, 3, 1, 1),                     # 2   conv1_2
            nn.ReLU(inplace = True),                        # 3   relu1_2
            nn.MaxPool2d(2, 2),                             # 4
            nn.Conv2d(64, 128, 3, 1, 1),                    # 5   conv2_1
            nn.ReLU(inplace = True),                        # 6   relu2_1
            nn.Conv2d(128, 128, 3, 1, 1),                   # 7   conv2_2
            nn.ReLU(inplace = True),                        # 8   relu2_2
            nn.MaxPool2d(2, 2),                             # 9
            nn.Conv2d(128, 256, 3, 1, 1),                   # 10  conv3_1
            nn.ReLU(inplace = True),                        # 11  relu3_1
            nn.Conv2d(256, 256, 3, 1, 1),                   # 12  conv3_2
            nn.ReLU(inplace = True),                        # 13  relu3_2
            nn.Conv2d(256, 256, 3, 1, 1),                   # 14  conv3_3
            nn.ReLU(inplace = True),                        # 15  relu3_3
            nn.MaxPool2d(2, 2),                             # 16
            nn.Conv2d(256, 512, 3, 1, 1),                   # 17  conv4_1
            nn.ReLU(inplace = True),                        # 18  relu4_1
            nn.Conv2d(512, 512, 3, 1, 1),                   # 19  conv4_2
            nn.ReLU(inplace = True),                        # 20  relu4_2
            nn.Conv2d(512, 512, 3, 1, 1),                   # 21  conv4_3
            nn.ReLU(inplace = True),                        # 22  relu4_3
            nn.MaxPool2d(2, 2),                             # 23
            nn.Conv2d(512, 512, 3, 1, 1),                   # 24  conv5_1
            nn.ReLU(inplace = True),                        # 25  relu5_1
            nn.Conv2d(512, 512, 3, 1, 1),                   # 26  conv5_2
            nn.ReLU(inplace = True),                        # 27  relu5_2
            nn.Conv2d(512, 512, 3, 1, 1),                   # 28  conv5_3
            nn.ReLU(inplace = True)                         # 29  relu5_3
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

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

def create_vggnet(vgg16_model_path):
    # Get the first 15 layers of vgg16, which is conv3_3
    vggnet = VGGFeaNet()
    # Pre-trained VGG-16
    vgg16 = torch.load(vgg16_model_path)
    load_dict(vggnet, vgg16)
    return vggnet
