import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups = groups, bias = False))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups = groups, bias = False)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

# ----------------------------------------
#            Dense-Res Block
# ----------------------------------------
class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False, scale_factor = 2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn),
            Conv2dLayer(latent_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, groups = groups, sn = sn)
        )
    
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out

class DenseConv2dLayer_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(DenseConv2dLayer_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv2 = Conv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv3 = Conv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv4 = Conv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv5 = Conv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
        
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv2 = Conv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv3 = Conv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv4 = Conv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv5 = Conv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = 0.1 * x5 + residual
        return x5

# ----------------------------------------
#            Group Conv Block
# ----------------------------------------
class DWConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(DWConv2dLayer, self).__init__()
        self.conv = nn.Sequential(
            # dw
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = in_channels, sn = sn),
            # pw
            Conv2dLayer(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = 'none', norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        return self.conv(x)

class DWTransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False, scale_factor = 2):
        super(DWTransposeConv2dLayer, self).__init__()
        self.conv = nn.Sequential(
            # dw
            TransposeConv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = in_channels, sn = sn, scale_factor = scale_factor),
            # pw
            Conv2dLayer(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = 'none', norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        return self.conv(x)

class InvertedResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(InvertedResidualLayer, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        # dw: depth-wise convolution
        # pw: point-wise convolution
        # pw-linear: point-wise convolution without activation
        self.conv = nn.Sequential(
            # pw
            Conv2dLayer(in_channels, hidden_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = activation, norm = norm, groups = 1, sn = sn),
            # dw
            Conv2dLayer(hidden_channels, hidden_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = hidden_channels, sn = sn),
            # pw-linear
            Conv2dLayer(hidden_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = 'none', norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        # dw: depth-wise convolution
        # pw: point-wise convolution
        # pw-linear: point-wise convolution without activation
        self.conv = nn.Sequential(
            # pw
            Conv2dLayer(in_channels, hidden_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = activation, norm = norm, groups = 1, sn = sn),
            # dw
            Conv2dLayer(hidden_channels, hidden_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = hidden_channels, sn = sn),
            # pw-linear
            Conv2dLayer(hidden_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = 'none', norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + 0.1 * self.conv(x)
        else:
            return self.conv(x)

class DWResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(DWResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = DWConv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv2 = DWConv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv3 = DWConv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv4 = DWConv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
        self.conv5 = DWConv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = 0.1 * x5 + residual
        return x5

# ----------------------------------------
#            ConvLSTM2d Block
# ----------------------------------------
class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad = False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad = False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# ----------------------------------------
#             Self Attn Block
# ----------------------------------------
class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_FM, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)
        
        out = self.gamma * out + x
        return out

class Self_Attn_C(nn.Module):
    """ Self attention Layer for Channel dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_C, self).__init__()
        self.chanel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.out_conv = nn.Conv2d(in_channels = in_dim // latent_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X c X c
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x c x c
        energy =  torch.bmm(proj_key, proj_query)
        # attention: B x c x c
        attention = self.softmax(energy)
        # proj_value is a convolution, B x c x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(batchsize, self.channel_latent, height, width)
        out = self.out_conv(out)
        
        out = self.gamma * out + x
        return out

if __name__ == "__main__":
    
    net = Conv2dLayer(in_channels = 3, out_channels = 3, kernel_size = 1, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 3, sn = False).cuda()
    a = torch.randn(1, 3, 64, 64).cuda()
    b = net(a)
    print(b.shape)
        
    net = DWConv2dLayer(in_channels = 3, out_channels = 3, kernel_size = 1, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 3, sn = False).cuda()
    a = torch.randn(1, 3, 64, 64).cuda()
    b = net(a)
    print(b.shape)
    