import torch
import torch.nn as nn
import torchvision.models as models
# from ResNet import ResNet50
from Res2Net_v1b import res2net50_v1b_26w_4s
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x




class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class DoubleAttention(nn.Module):
    def __init__(
        self,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(DoubleAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial


    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_cat = torch.max(x_out11,x_out21)
        x_out = x_cat*x


        return x_out






class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()

        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                #nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class multi_scale_aspp(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, channel=32):
        super(multi_scale_aspp, self).__init__()
        self.ASPP_3 = _DenseAsppBlock(input_num=channel, num1=channel * 2, num2=channel, dilation_rate=3, drop_out=0.1, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=channel * 2, num1=channel * 2, num2=channel,
                                      dilation_rate=6, drop_out=0.1, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=channel * 3, num1=channel * 2, num2=channel,
                                       dilation_rate=12, drop_out=0.1, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=channel * 4, num1=channel * 2, num2=channel,
                                       dilation_rate=18, drop_out=0.1, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=channel * 5, num1=channel * 2, num2=channel,
                                       dilation_rate=24, drop_out=0.1, bn_start=True)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 6, out_channels=channel, kernel_size=1, padding=0)
        )

    def forward(self, _input):
        # feature = super(_DenseAsppBlock, self).forward(_input)
        aspp3 = self.ASPP_3(_input)
        feature = torch.cat((aspp3, _input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)

        feature = torch.cat((aspp24, feature), dim=1)

        aspp_feat = self.classification(feature)

        return aspp_feat

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)



#Network
class Network(nn.Module):
    def __init__(self, channel=32,imagenet_pretrained=True,mode=True):
        super(Network, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained,mode='rgb')

        # self.resnet = ResNet50('rgb')
        self.resnet_depth=res2net50_v1b_26w_4s(pretrained=imagenet_pretrained,mode='rgb')

        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv96to32 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2048to32 = nn.Conv2d(2048, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1024to32 = nn.Conv2d(1024, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv512to32 = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu_1 = nn.PReLU()
        self.conv_1 = nn.Conv2d(32, 1, 1, padding=0)

        #upsample function
        self.upsample1_1 = nn.Sequential( nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False), nn.BatchNorm2d(32),nn.PReLU())
        self.upsample1_2 = nn.Sequential( nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=False), nn.BatchNorm2d(32),nn.PReLU())
        self.upsample1_3 = nn.Sequential( nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False), nn.BatchNorm2d(32),nn.PReLU())
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        

        # MultiAttention
        self.DoubleAttention2 = DoubleAttention()
        self.DoubleAttention3 = DoubleAttention()
        self.DoubleAttention4 = DoubleAttention()



        # multi_scale_aspp
        self.multi_scale_aspp4 = multi_scale_aspp(channel)
        self.multi_scale_aspp3 = multi_scale_aspp(channel)
        self.multi_scale_aspp2 = multi_scale_aspp(channel)
       

    def forward(self, x, x_depth):
        # ==========================  res2net  ==========================
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
	    #将深度通道变为3
        #x_depth = self.conv1to3(x_depth)
        x_depth = torch.cat((x_depth,x_depth,x_depth),1)
        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)
        # layer1
        x1 = self.resnet.layer1(x)  # 256 x 88 x 88
        x1_depth=self.resnet_depth.layer1(x_depth)
        # layer2
        x2 = self.resnet.layer2(x1)  # 512 x 44 x 44
        x2_depth=self.resnet_depth.layer2(x1_depth)
        # layer3
        x3 = self.resnet.layer3(x2)  # 1024 x 22 x 22
        x3_depth = self.resnet_depth.layer3(x2_depth)  # 1024 x 22 x 22
        # layer 4
        x4 = self.resnet.layer4(x3)  # 2048 x 11 x 11
        x4_depth = self.resnet_depth.layer4(x3_depth)

      


        x2_1 = x2 + x2_depth
        x3_1 = x3 + x3_depth
        x4_1 = x4 + x4_depth


        f4 = self.conv2048to32(x4_1)
        f3 = self.conv1024to32(x3_1)
        f2 = self.conv512to32(x2_1)

        f4_1 = self.multi_scale_aspp4(f4)
        f3_1 = self.multi_scale_aspp3(f3) 
        f2_1 = self.multi_scale_aspp2(f2)

        f4_2 = torch.sigmoid(self.upsample2(f4_1))
        f3_2 = torch.sigmoid(self.upsample2(f3_1))
        f3_3 = f3_1 * f4_2
        f3_4 = f3_1 + f3_3 + self.upsample2(f4_1)
        f3_5 = torch.sigmoid(self.upsample2(f3_4))
        f2_2 = f2_1 * f3_2
        f2_3 = f2_1 + f2_2
        f2_4 = f2_3 * f3_5
        f2_5 = f2_3 + f2_4 + self.upsample2(f3_4)




       
        #
        #x2_4 = torch.cat((x2_3,self.upsample2(x3_3),self.upsample4(x4_3)),1)



        y = self.upsample1_1(self.upsample4(f2_5)) #
        y = self.relu_1(self.conv_1(y))  # 1 352 352


        return y



