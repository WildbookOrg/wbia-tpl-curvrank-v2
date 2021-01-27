# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # resnet50 = models.resnet50(pretrained=True)
        resnet50 = models.resnet50(pretrained=False)

        self.resnet50 = nn.Sequential(*list(resnet50.children())[0:-2])
        # Need to include a new average pooling layer because the pretrained
        # one had its dimensions set for an input size of (224, 224).
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.predict_start = nn.Linear(512 * 4, 2)
        self.predict_end = nn.Linear(512 * 4, 2)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x0 = self.predict_start(x)
        x1 = self.predict_end(x)

        return x0, x1


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # vgg = models.vgg16_bn(pretrained=True)
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.regressor = nn.Sequential(*list(vgg.classifier.children())[0:-1])

        self.predict_start = nn.Linear(4096, 2)
        self.predict_end = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x0 = self.predict_start(x)
        x1 = self.predict_end(x)

        return x0, x1


if __name__ == '__main__':
    # model = ResNet50()
    model = VGG16()
