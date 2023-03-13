import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_module import *
import networks.network_utils as nutils
import datasets.data_utils as dutils

# ----------------------------------------
#        Correspondence-based fusion
# ----------------------------------------
class Correspondence_Net(nn.Module):
    def __init__(self, opt):
        super(Correspondence_Net, self).__init__()
        
        self.opt = opt
        self.vggnet = VGG19_pytorch()
        self.nonlocal_net = WarpNet(1)
    
    def feature_normalize(self, feature_in):
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim = True) + sys.float_info.epsilon
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

    def forward(self, IA_l, IB_lab, features_B, temperature = 0.01):
        # A is the grayscale image at the time step t
        # B is the colorized frame at the first time 1
        IA_rgb_from_gray = nutils.gray2rgb_batch(IA_l)
        with torch.no_grad():
            A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = self.vggnet(
                IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess = True
            )
            B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

        # NOTE: output the feature before normalization
        features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

        A_relu2_1 = self.feature_normalize(A_relu2_1)
        A_relu3_1 = self.feature_normalize(A_relu3_1)
        A_relu4_1 = self.feature_normalize(A_relu4_1)
        A_relu5_1 = self.feature_normalize(A_relu5_1)
        B_relu2_1 = self.feature_normalize(B_relu2_1)
        B_relu3_1 = self.feature_normalize(B_relu3_1)
        B_relu4_1 = self.feature_normalize(B_relu4_1)
        B_relu5_1 = self.feature_normalize(B_relu5_1)

        nonlocal_BA_lab, similarity_map = self.nonlocal_net(
            IB_lab,
            A_relu2_1,
            A_relu3_1,
            A_relu4_1,
            A_relu5_1,
            B_relu2_1,
            B_relu3_1,
            B_relu4_1,
            B_relu5_1,
            temperature = temperature,
        )

        # WLS (guided filter)
        nonlocal_BA_l = nonlocal_BA_lab[:, [0], :, :]
        guide_image = dutils.uncenter_l(IA_l) * 255 / 100
        wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), self.opt.lambda_value, self.opt.sigma_color
        )
        curr_predict_a = wls_filter.filter(nonlocal_BA_lab[0, 1, :, :].cpu().numpy())
        curr_predict_b = wls_filter.filter(nonlocal_BA_lab[0, 2, :, :].cpu().numpy())
        curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0).cuda()
        curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0).cuda()
        nonlocal_BA_lab = torch.cat((nonlocal_BA_l, curr_predict_a, curr_predict_b), dim = 1)

        # Convert color space to fit CombinationNet
        nonlocal_BA_lab_to_cpnet_rgb = nutils.corr_lab_to_cv2_rgb(nonlocal_BA_lab, IA_l)
        nonlocal_BA_lab_to_cpnet_ab = nutils.cv2_rgb_to_tensor_ab(nonlocal_BA_lab_to_cpnet_rgb).cuda()

        return nonlocal_BA_lab_to_cpnet_ab, similarity_map

# ----------------------------------------
#     Pre-defined networks from DEVC
# ----------------------------------------
def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    return tensor_bgr_ml * 255

class VGG19_pytorch(nn.Module):
    """
    NOTE: no need to pre-process the input; input tensor should range in [0,1]
    """
    def __init__(self, pool="max"):
        super(VGG19_pytorch, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        """
        NOTE: input tensor should range in [0,1]
        """
        out = {}
        if preprocess:
            x = vgg_preprocess(x)
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out

class WarpNet(nn.Module):
    """ input is Al, Bl, channel = 1, range~[0,255] """

    def __init__(self, batch_size):
        super(WarpNet, self).__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
        )

        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        B_lab_map,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=0.001 * 5,
        detach_flag=False,
        WTA_scale_weight=1,
        feature_noise=0,
    ):
        batch_size = B_lab_map.shape[0]
        channel = B_lab_map.shape[1]
        image_height = B_lab_map.shape[2]
        image_width = B_lab_map.shape[3]
        feature_height = int(image_height / 4)
        feature_width = int(image_width / 4)

        # scale feature size to 44*44
        A_feature2_1 = self.layer2_1(A_relu2_1)
        B_feature2_1 = self.layer2_1(B_relu2_1)
        A_feature3_1 = self.layer3_1(A_relu3_1)
        B_feature3_1 = self.layer3_1(B_relu3_1)
        A_feature4_1 = self.layer4_1(A_relu4_1)
        B_feature4_1 = self.layer4_1(B_relu4_1)
        A_feature5_1 = self.layer5_1(A_relu5_1)
        B_feature5_1 = self.layer5_1(B_relu5_1)

        # concatenate features
        if A_feature5_1.shape[2] != A_feature2_1.shape[2] or A_feature5_1.shape[3] != A_feature2_1.shape[3]:
            A_feature5_1 = F.pad(A_feature5_1, (0, 0, 1, 1), "replicate")
            B_feature5_1 = F.pad(B_feature5_1, (0, 0, 1, 1), "replicate")
        A_features = self.layer(torch.cat((A_feature2_1, A_feature3_1, A_feature4_1, A_feature5_1), 1))
        B_features = self.layer(torch.cat((B_feature2_1, B_feature3_1, B_feature4_1, B_feature5_1), 1))

        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(B_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

        # downsample the reference color
        B_lab = F.avg_pool2d(B_lab_map, 4)
        B_lab = B_lab.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)  # 2*1936*channel

        # multiply the corr map with color
        y = torch.matmul(f_div_C, B_lab)  # 2*1936*channel
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        y = self.upsampling(y)
        similarity_map = self.upsampling(similarity_map)

        return y, similarity_map
