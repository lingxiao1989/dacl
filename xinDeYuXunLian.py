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
from resnet import BasicBlock, Bottleneck, ResNet, resnet18

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

        self.feat_out1 = nn.Linear(200704, num_classes)
        self.feat_out2 = nn.Linear(50176, num_classes)
        self.feat_out3 = nn.Linear(12544, num_classes)
        self.feat_out4 = nn.Linear(4096, num_classes)
        self.feat_out5 = nn.Linear(1024, num_classes)

        self.final_out = nn.Linear(num_classes*5, num_classes)
            # print(self.features)
        # self.features.add_module("final_pool", nn.AvgPool2d(
        #     kernel_size=7,
        #     stride=1))
        self.final_pool =  nn.AvgPool2d(kernel_size=7,stride=1)
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
        print("c2.shape==================", c2.shape)
        print("c3.shape==================", c3.shape)
        print("c4.shape==================", c4.shape)
        print("c5.shape==================", c5.shape)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        print("p3.shape==================", p3.shape)
        print("p4.shape==================", p4.shape)
        print("p5.shape==================", p5.shape)
        print("p6.shape==================", p6.shape)
        print("p7.shape==================", p7.shape)

        '''keras implementation
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
        P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

        P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
        P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
        P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

        # Concatenate P5 elementwise to C4
        P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
        P4 = keras.layers.Concatenate(axis=3)([P5_upsampled, P4])
        P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
        P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P4')(P4)

        # Concatenate P4 elementwise to C3
        P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
        P3 = keras.layers.Concatenate(axis=3)([P4_upsampled, P3])
        P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P3')(P3)
        '''
        # x = self.avgpool(x)
        # print("x.shape==================", x.shape)

        feature1 = torch.flatten(p3, 1)
        feature1 = torch.nn.Dropout(0.5)(feature1)
        feature1 = self.feat_out1(feature1)

        print("feature1.shape==================", feature1.shape)
        '''
        feature2 = torch.flatten(p4, 1)
        feature2 = torch.nn.Dropout(0.5)(feature2)
        feature2 = self.feat_out2(feature2)

        print("feature2.shape==================", feature2.shape)

        feature3 = torch.flatten(p5, 1)
        feature3 = torch.nn.Dropout(0.5)(feature3)
        feature3 = self.feat_out3(feature3)

        feature4 = torch.flatten(p6, 1)
        feature4 = torch.nn.Dropout(0.5)(feature4)
        feature4 = self.feat_out4(feature4)

        feature5 = torch.flatten(p7, 1)
        feature5 = torch.nn.Dropout(0.5)(feature5)
        feature5 = self.feat_out5(feature5)

        concat = torch.cat((feature1,feature2,feature3,feature4,feature5),1)

        print("concat.shape==================", concat.shape)

        out=self.final_out(concat)
        keras implementation
        # Run classification for each of the generated features from the pyramid
        feature1 = Flatten()(P3)
        dp1 = Dropout(0.5)(feature1)
        preds1 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp1)
        #################################################################
        feature2 = Flatten()(P4)
        dp2 = Dropout(0.5)(feature2)
        preds2 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp2)
        #################################################################
        feature3 = Flatten()(P5)
        dp3= Dropout(0.5)(feature3)
        preds3 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp3)
        #################################################################
        feature4 = Flatten()(P6)
        dp4 = Dropout(0.5)(feature4)
        preds4 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp4)
        #################################################################
        feature5 = Flatten()(P7)
        dp5 = Dropout(0.5)(feature5)
        preds5 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp5)
        #################################################################
        concat=keras.layers.Concatenate(axis=1)([preds1,preds2,preds3,preds4,preds5]) #Concatenate the predictions(Classification results) of each of the pyramid features 
        out=keras.layers.Dense(2,activation='softmax',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(concat) #Final Classification

        model = Model(inputs=base_model.input, outputs=out) #Create the Training Model
        '''
        '''
        # DACL attention
        x_flat = torch.flatten(x, 1)
        # print("x_flat.shape==================", x.shape)
        E = self.attention(x_flat)
        # print("E.shape==================", E.shape)
        A = self.attention_heads(E).reshape(-1, 2048, 2).softmax(dim=-1)[:, :, 1]
        # print("A.shape==================", A.shape)
        '''

        #x = self.avgpool(x)
        #f = torch.flatten(x, 1)
        # f = A.view(A.size(0), -1)
        # print("f.shape==================", f.shape)
        #out = self.fc(f)
        # out = self.output(f)
        # print("out.shape==================", out.shape)

        #return f, out, A
        #x = self.final_pool(c5)
        #x = x.view(x.size(0), -1)
        #out = self.output(x)
        return feature1

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

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUnit, self).__init__()
        width = int(out_channels / 4)

        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # for downsample
        self._downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self._downsample(identity)
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BaseNet(nn.Module):
    """basenet for fer2013"""

    def __init__(self, in_channels=1, num_classes=7):
        super(BaseNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_1 = ResidualUnit(in_channels=64, out_channels=256)
        self.residual_2 = ResidualUnit(in_channels=256, out_channels=512)
        self.residual_3 = ResidualUnit(in_channels=512, out_channels=1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def basenet(in_channels=1, num_classes=7):
    return BaseNet(in_channels, num_classes)


from masking import masking


class ResMasking(ResNet):
    def __init__(self, weight_path):
        super(ResMasking, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000
        )
        # state_dict = torch.load('saved/checkpoints/resnet18_rot30_2019Nov05_17.44')['net']
        # state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        # self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        """
        # freeze all net
        for m in self.parameters():
            m.requires_grad = False
        """

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)
        
        self.nb_head = 512
        self.attention = nn.Sequential(
            #nn.Linear(2048 * 7 * 7, 512),
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

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56
        m = self.mask1(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer2(x)  # 28
        m = self.mask2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14
        m = self.mask3(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer4(x)  # 7
        m = self.mask4(x)
        x = x * (1 + m)
        # x = x * m

        # DACL attention
        x_flat = torch.flatten(x, 1)
        # print("x_flat.shape==================", x.shape)
        E = self.attention(x_flat)
        # print("E.shape==================", E.shape)
        A = self.attention_heads(E).reshape(-1, 512, 2).softmax(dim=-1)[:, :, 1]
        # print("A.shape==================", A.shape)


        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        # f = A.view(A.size(0), -1)
        # print("f.shape==================", f.shape)
        out = self.fc(f)
        # out = self.output(f)
        # print("out.shape==================", out.shape)

        return f, out, A

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        #return x


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
        #cbam_resnet50,
        # cbam_resnet101,
        # cbam_resnet152,
        basenet
    ]

    for model in models:
        #net = model(pretrained=pretrained)
        net = model()
        print(net)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != cbam_resnet18 or weight_count == 11779392)
        # assert (model != cbam_resnet34 or weight_count == 21960468)
        # assert (model != cbam_resnet50 or weight_count == 28089624)
        # assert (model != cbam_resnet101 or weight_count == 49330172)
        # assert (model != cbam_resnet152 or weight_count == 66826848)

        x = torch.randn(1, 1, 224, 224)
        y = net(x)
        #y.sum().backward()
        #assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

