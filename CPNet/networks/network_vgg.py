import torch
import torch.nn as nn

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

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
def create_vggnet(vgg16_model_path):
    # Get the first 15 layers of vgg16, which is conv3_3
    vggnet = VGGFeaNet()
    # Pre-trained VGG-16
    vgg16 = torch.load(vgg16_model_path)
    load_dict(vggnet, vgg16)
    return vggnet

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

# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self, vgg_name):
        super(PerceptualNet, self).__init__()
        self.vgg = create_vggnet(vgg_name)
        self.conv1_2 = []
        self.conv2_2 = []
        self.conv3_3 = []
        self.conv4_3 = []
        self.conv5_3 = []
        for i in range(0, 4):
            self.conv1_2.append(self.vgg.features[i])
        for i in range(4, 9):
            self.conv2_2.append(self.vgg.features[i])
        for i in range(9, 16):
            self.conv3_3.append(self.vgg.features[i])
        for i in range(16, 23):
            self.conv4_3.append(self.vgg.features[i])
        for i in range(23, 29):
            self.conv5_3.append(self.vgg.features[i])
        self.conv1_2 = nn.Sequential(*self.conv1_2)
        self.conv2_2 = nn.Sequential(*self.conv2_2)
        self.conv3_3 = nn.Sequential(*self.conv3_3)
        self.conv4_3 = nn.Sequential(*self.conv4_3)
        self.conv5_3 = nn.Sequential(*self.conv5_3)
    
    def forward(self, x):
        # vgg encoder
        x1 = self.conv1_2(x)                                        # out: batch * 64 * 256 * 256
        x2 = self.conv2_2(x1)                                       # out: batch * 128 * 128 * 128
        x3 = self.conv3_3(x2)                                       # out: batch * 256 * 64 * 64
        x4 = self.conv4_3(x3)                                       # out: batch * 512 * 32 * 32
        x5 = self.conv5_3(x4)                                       # out: batch * 512 * 16 * 16
        return x1, x2, x3, x4, x5

if __name__ == '__main__':

    def create_perceptualnet(vgg16_model_path):
        # Get the first 15 layers of vgg16, which is conv3_3
        perceptualnet = PerceptualNet()
        # Pre-trained VGG-16
        vgg16 = torch.load(vgg16_model_path)
        load_dict(perceptualnet, vgg16)
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

    '''
    net = create_perceptualnet('util/vgg16_pretrained.pth').cuda()
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)
    '''

    vgg = VGGFeaNet().cuda()
    conv1_2 = []
    conv2_2 = []
    conv3_2 = []
    conv4_3 = []
    conv5_3 = []
    for i in range(0, 4):
        conv1_2.append(vgg.features[i])
    for i in range(4, 9):
        conv2_2.append(vgg.features[i])
    for i in range(9, 16):
        conv3_2.append(vgg.features[i])
    for i in range(16, 23):
        conv4_3.append(vgg.features[i])
    for i in range(23, 29):
        conv5_3.append(vgg.features[i])
    conv1_2 = nn.Sequential(*conv1_2)
    conv2_2 = nn.Sequential(*conv2_2)
    conv3_2 = nn.Sequential(*conv3_2)
    conv4_3 = nn.Sequential(*conv4_3)
    conv5_3 = nn.Sequential(*conv5_3)
    print(conv1_2)
    print(conv2_2)
    print(conv3_2)
    print(conv4_3)
    print(conv5_3)

    a = torch.randn(1, 3, 256, 256).cuda()
    conv1_2 = conv1_2(a)
    conv2_2 = conv2_2(conv1_2)
    conv3_2 = conv3_2(conv2_2)
    conv4_3 = conv4_3(conv3_2)
    conv5_3 = conv5_3(conv4_3)
    print(a.shape, conv1_2.shape, conv2_2.shape, conv3_2.shape, conv4_3.shape, conv5_3.shape)
