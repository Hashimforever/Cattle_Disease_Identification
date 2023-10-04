import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from ccsam_module import CCSAM
from torch import nn
from torchvision.models.inception import Inception3
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(BasicBlock, self).__init__()
       
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ccsam = CCSAM(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ccsam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CCSAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(CCSAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ccsam = CCSAM(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ccsam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet18_ccsam(num_classes=5):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, 256)
   
    # Add a dropout layer with probability 0.5
    model.dropout = nn.Dropout(p=0.5)

   
    # Add a new fully connected layer with 5 output classes and softmax activation
    model.fc2 = nn.Linear(256, 5)
    model.softmax = nn.Softmax(dim=1)

    return model

#model=resnet18_ccsam(num_classes=5)
#print("this model is Fine_tuned ResNet18 with Combination of channel Attention and Spatial Attention Mechanism")
#print(model)

class CCSAMInception3(nn.Module):
    def __init__(self, num_classes=5, aux_logits=True, transform_input=False):
        super(CCSAMInception3, self).__init__()
        model = Inception3(num_classes=num_classes, aux_logits=aux_logits,
                           transform_input=transform_input)
        model.Mixed_5b.add_module("CCSAM", CCSAM(192))
        model.Mixed_5c.add_module("CCSAM", CCSAM(256))
        model.Mixed_5d.add_module("CCSAM", CCSAM(288))
        model.Mixed_6a.add_module("CCSAM", CCSAM(288))
        model.Mixed_6b.add_module("CCSAM", CCSAM(768))
        model.Mixed_6c.add_module("CCSAM", CCSAM(768))
        model.Mixed_6d.add_module("CCSAM", CCSAM(768))
        model.Mixed_6e.add_module("CCSAM", CCSAM(768))
        if aux_logits:
            model.AuxLogits.add_module("CCSAM", CCSAM(768))
        model.Mixed_7a.add_module("CCSAM", CCSAM(768))
        model.Mixed_7b.add_module("CCSAM", CCSAM(1280))
        model.Mixed_7c.add_module("CCSAM", CCSAM(2048))

        self.model = model

    def forward(self, x):
        _, _, h, w = x.size()
        if (h, w) != (299, 299):
            raise ValueError("input size must be (299, 299)")

        return self.model(x)


def ccsam_inception_v3(**kwargs):
    return CCSAMInception3(**kwargs)
#model=ccsam_inception_v3(num_classes=5)
#print("this model is inception_v3 with Combination of channel Attention and Spatial Attention Mechanism")
#print(model)

class Freezing_layer_Resnet18WithCCSAM(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(Freezing_layer_Resnet18WithCCSAM, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        for name, param in self.model.named_parameters():
            if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
                param.requires_grad = False

        self.model.layer1[0].add_module("CCSAM", CCSAM(64))
        self.model.layer1[1].add_module("CCSAM", CCSAM(64))
        self.model.layer2[0].add_module("CCSAM", CCSAM(128))
        self.model.layer2[1].add_module("CCSAM", CCSAM(128))
        self.model.layer3[0].add_module("CCSAM", CCSAM(256))
        self.model.layer3[1].add_module("CCSAM", CCSAM(256))
        self.model.layer4[0].add_module("CCSAM", CCSAM(512))
        self.model.layer4[1].add_module("CCSAM", CCSAM(512))

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)
        self.model.dropout = nn.Dropout(p=0.5)
        self.model.fc2 = nn.Linear(256, num_classes)
        self.model.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.model(x)
        return out
def Freezing_layer_Resnet18WithCCSAM1(num_classes=5):
    return Freezing_layer_Resnet18WithCCSAM(num_classes=5)
#model=Freezing_layer_Resnet18WithCCSAM1(num_classes=5)
#print("this model is Freezing_layer_Resnet18 with Combination of channel Attention and Spatial Attention Mechanism")
#$print(model)
