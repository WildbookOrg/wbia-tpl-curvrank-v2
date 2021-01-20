# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(ResNet34, self).__init__()
        # resnet34 = models.resnet34(pretrained=True)
        resnet34 = models.resnet34()
        if pretrained_weights is not None:
            resnet34.load_state_dict(torch.load(pretrained_weights))
        self.resnet34 = nn.Sequential(*list(resnet34.children())[0:-1])
        # Need to include a new average pooling layer because the pretrained
        # one had its dimensions set for an input size of (224, 224).
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.predict_start = nn.Linear(512, 2)
        self.predict_end = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet34(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x0 = self.predict_start(x)
        x1 = self.predict_end(x)

        return x0, x1


class ResNet50(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(ResNet50, self).__init__()
        # resnet50 = models.resnet50(pretrained=True)
        resnet50 = models.resnet50()
        if pretrained_weights is not None:
            resnet50.load_state_dict(torch.load(pretrained_weights))
        self.resnet50 = nn.Sequential(*list(resnet50.children())[0:-1])
        # Need to include a new average pooling layer because the pretrained
        # one had its dimensions set for an input size of (224, 224).
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.predict_start = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 2)
        )
        self.predict_end = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        x0 = self.predict_start(x)
        x1 = self.predict_end(x)

        return x0, x1


class VGG16(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(VGG16, self).__init__()
        # vgg = models.vgg16_bn(pretrained=True)
        vgg = models.vgg16()
        if pretrained_weights is not None:
            vgg.load_state_dict(torch.load(pretrained_weights))
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


class VGG16Coarse(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(VGG16, self).__init__()
        # vgg = models.vgg16_bn(pretrained=True)
        vgg = models.vgg16()
        if pretrained_weights is not None:
            vgg.load_state_dict(torch.load(pretrained_weights))
        self.features = vgg.features
        self.regressor = nn.Sequential(*list(vgg.classifier.children())[0:-1])

        self.adapt = nn.Sequential(nn.Conv2d(4, 3, kernel_size=3, padding=1), nn.ReLU())

        self.predict_start = nn.Linear(4096, 2)
        self.predict_end = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.adapt(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x0 = self.predict_start(x)
        x1 = self.predict_end(x)

        return x0, x1


class VGG19(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(VGG19, self).__init__()
        # vgg = models.vgg19_bn(pretrained=True)
        vgg = models.vgg19_bn()
        if pretrained_weights is not None:
            vgg.load_state_dict(torch.load(pretrained_weights))
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
