"""
    加了预训练
"""
__all__ = ['CbamResNet',
           # 'cbam_resnet18', 'cbam_resnet34',
           'cbam_resnet50',
           # 'cbam_resnet101', 'cbam_resnet152'
           ]

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from pytorchcv.models.common import conv1x1_block, conv7x7_block
from pytorchcv.models.resnet import ResInitBlock, ResBlock, ResBottleneck
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multilayer perceptron block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(
            in_features=channels,
            out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_channels,
            out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x

class ChannelGate(nn.Module):
    """
    CBAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(ChannelGate, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        return x

class SpatialGate(nn.Module):
    """
    CBAM spatial gate block.
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = conv7x7_block(
            in_channels=2,
            out_channels=1,
            activation=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = x.max(dim=1)[0].unsqueeze(1)
        att2 = x.mean(dim=1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.conv(att)
        att = self.sigmoid(att)
        x = x * att
        return x

class CbamBlock(nn.Module):
    """
    CBAM attention block for CBAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(CbamBlock, self).__init__()
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate()

    def forward(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x

class CbamResUnit(nn.Module):
    """
    CBAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(CbamResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=False)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.cbam = CbamBlock(channels=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.cbam(x)
        x = x + identity
        x = self.activ(x)
        return x

class CbamResNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(CbamResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), CbamResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
            # print(self.features)
        # self.features.add_module("final_pool", nn.AvgPool2d(
        #     kernel_size=7,
        #     stride=1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # DACL attention network
        self.nb_head = 2048
        self.attention = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Linear(3584, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
        )
        self.attention_heads = nn.Linear(64, 2 * self.nb_head)

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)
        self.fc = nn.Linear(2048, num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        # print("input.shape==================",x.shape)
        x = self.features(x)
        # print("feature.shape==================",x.shape)
        # x = self.avgpool(x)
        # print("x.shape==================", x.shape)

        # DACL attention
        x_flat = torch.flatten(x, 1)
        # print("x_flat.shape==================", x.shape)
        E = self.attention(x_flat)
        # print("E.shape==================", E.shape)
        A = self.attention_heads(E).reshape(-1, 2048, 2).softmax(dim=-1)[:, :, 1]
        # print("A.shape==================", A.shape)


        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        # f = A.view(A.size(0), -1)
        # print("f.shape==================", f.shape)
        out = self.fc(f)
        # out = self.output(f)
        # print("out.shape==================", out.shape)

        return f, out, A
        # return out 改成下面
        # x = x.view(x.size(0), -1)
        # x = self.output(x)
        # return out

class FPNCbamResNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(FPNCbamResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        ##FPN layers
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        #self.layer2 = self._make_layer(ResInitBlock,  64, channels[0][0], stride=1) (stage1)
        #self.layer3 = self._make_layer(ResInitBlock, 128, channels[1][0], stride=2) (stage2)
        #self.layer4 = self._make_layer(ResInitBlock, 256, channels[2][0], stride=2) (stage3)
        #self.layer5 = self._make_layer(ResInitBlock, 512, channels[3][0], stride=2) (stage4)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        ###

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        self.stage1 = nn.Sequential()
        for j, out_channels in enumerate(channels[0]):
            stride = 2 if (j == 0) and (0 != 0) else 1
            self.stage1.add_module("unit{}".format(j + 1), CbamResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bottleneck=bottleneck))
            in_channels = out_channels
        self.stage2 = nn.Sequential()
        for j, out_channels in enumerate(channels[1]):
            stride = 2 if (j == 0) and (1 != 0) else 1
            self.stage2.add_module("unit{}".format(j + 1), CbamResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bottleneck=bottleneck))
            in_channels = out_channels
        self.stage3 = nn.Sequential()
        for j, out_channels in enumerate(channels[2]):
            stride = 2 if (j == 0) and (2 != 0) else 1
            self.stage3.add_module("unit{}".format(j + 1), CbamResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bottleneck=bottleneck))
            in_channels = out_channels
        self.stage4 = nn.Sequential()
        for j, out_channels in enumerate(channels[3]):
            stride = 2 if (j == 0) and (3 != 0) else 1
            self.stage4.add_module("unit{}".format(j + 1), CbamResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bottleneck=bottleneck))
            in_channels = out_channels


            # print(self.features)
        # self.features.add_module("final_pool", nn.AvgPool2d(
        #     kernel_size=7,
        #     stride=1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # DACL attention network
        self.nb_head = 2048
        self.attention = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Linear(3584, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
        )
        self.attention_heads = nn.Linear(64, 2 * self.nb_head)

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)
        self.fc = nn.Linear(2048, num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    '''
    def forward(self, x):
        # Bottom-up
        #resinitblock
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7

    '''

    def forward(self, x):
        # print("input.shape==================",x.shape)
        x = self.features(x)
        # print("feature.shape==================",x.shape)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)        

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)

        # x = self.avgpool(x)
        # print("x.shape==================", x.shape)

        # DACL attention
        x_flat = torch.flatten(x, 1)
        # print("x_flat.shape==================", x.shape)
        E = self.attention(x_flat)
        # print("E.shape==================", E.shape)
        A = self.attention_heads(E).reshape(-1, 2048, 2).softmax(dim=-1)[:, :, 1]
        # print("A.shape==================", A.shape)


        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        # f = A.view(A.size(0), -1)
        # print("f.shape==================", f.shape)
        out = self.fc(f)
        # out = self.output(f)
        # print("out.shape==================", out.shape)

        return f, out, A
        # return out 改成下面
        # x = x.view(x.size(0), -1)
        # x = self.output(x)
        # return out

def get_resnet(blocks,model_name=None,pretrained=False,root=os.path.join("~", ".torch", "models"),**kwargs):
    """
    Create CBAM-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported CBAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    #net = CbamResNet(
    print(channels)
    net = FPNCbamResNet(        
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        # from pytorchcv.models.model_store import download_model
        # download_model(
        #     net=net,
        #     model_name=model_name,
        #     local_model_store_dir_path=root)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        model_dict = net.state_dict()

        # 将pretrained_dict里不属于model_dict的键剔除掉
        pre_dict = torch.load("/home/zhimahu/jjy/ResidualMaskingNetwork-master/cbam_resnet50_rot30_2019Nov15_12.40")["net"]

        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}

        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print("++++++++++===================================================================================================================")
    return net

# def cbam_resnet18(**kwargs):
#     """
#     CBAM-ResNet-18 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.torch/models'
#         Location for keeping the model parameters.
#     """
#     return get_resnet(blocks=18, model_name="cbam_resnet18", **kwargs)
#
#
# def cbam_resnet34(**kwargs):
#     """
#     CBAM-ResNet-34 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.torch/models'
#         Location for keeping the model parameters.
#     """
#     return get_resnet(blocks=34, model_name="cbam_resnet34", **kwargs)

def cbam_resnet50(**kwargs):
    """
    CBAM-ResNet-50 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="cbam_resnet50" ,pretrained=False , **kwargs)

# def cbam_resnet101(**kwargs):
#     """
#     CBAM-ResNet-101 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.torch/models'
#         Location for keeping the model parameters.
#     """
#     return get_resnet(blocks=101, model_name="cbam_resnet101", **kwargs)
#
#
# def cbam_resnet152(**kwargs):
#     """
#     CBAM-ResNet-152 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     root : str, default '~/.torch/models'
#         Location for keeping the model parameters.
#     """
#     return get_resnet(blocks=152, model_name="cbam_resnet152", **kwargs)

def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count

def _test():
    import torch

    #pretrained = False

    models = [
        # cbam_resnet18,
        # cbam_resnet34,
        cbam_resnet50,
        # cbam_resnet101,
        # cbam_resnet152,
    ]

    for model in models:
        #net = model(pretrained=pretrained)
        net = model()
        print(net)
'''
        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != cbam_resnet18 or weight_count == 11779392)
        # assert (model != cbam_resnet34 or weight_count == 21960468)
        # assert (model != cbam_resnet50 or weight_count == 28089624)
        # assert (model != cbam_resnet101 or weight_count == 49330172)
        # assert (model != cbam_resnet152 or weight_count == 66826848)

        x = torch.randn(1, 3, 224, 224)
        f,out,a  = net(x)
        out.sum().backward()
        assert (tuple(out.size()) == (1, 1000))
'''

if __name__ == "__main__":
    _test()

