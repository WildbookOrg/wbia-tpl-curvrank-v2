# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if isinstance(m.kernel_size, tuple):
            N = np.prod(m.kernel_size) * m.in_channels
        else:
            N = m.kernel_size * m.kernel_size * m.in_channels
        mean = np.sqrt(2.0 / N)
        nn.init.normal_(m.weight, mean=mean)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Parameters for the contracting convolutional layers.
        con_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        # Parameters for the expanding convolutional layers.
        exp_params = {'kernel_size': 2, 'stride': 2, 'padding': 0}
        # The output layer uses 1x1 convolution.
        out_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.con1 = nn.Sequential(
            nn.Conv2d(3, 64, **con_params),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, **con_params),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # self.con1.apply(init_weights)

        self.con2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, **con_params),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, **con_params),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # self.con2.apply(init_weights)

        self.con3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, **con_params),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, **con_params),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # self.con3.apply(init_weights)

        self.con4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, **con_params),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, **con_params),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # self.con4.apply(init_weights)

        self.con5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, **con_params),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, **con_params),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, **exp_params),
        )
        # self.con5.apply(init_weights)

        self.exp1 = nn.Sequential(
            nn.Conv2d(1024, 512, **con_params),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, **con_params),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, **exp_params),
        )
        # self.exp1.apply(init_weights)

        self.exp2 = nn.Sequential(
            nn.Conv2d(512, 256, **con_params),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, **con_params),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, **exp_params),
        )
        # self.exp2.apply(init_weights)

        self.exp3 = nn.Sequential(
            nn.Conv2d(256, 128, **con_params),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, **con_params),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, **exp_params),
        )
        # self.exp3.apply(init_weights)

        self.exp4 = nn.Sequential(
            nn.Conv2d(128, 64, **con_params),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, **con_params),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, **out_params),
        )
        # self.exp4.apply(init_weights)

        self.predict_proba = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.con1(x)
        x2 = self.con2(x1)
        x3 = self.con3(x2)
        x4 = self.con4(x3)
        x5 = self.con5(x4)

        x6 = self.exp1(torch.cat((x4, x5), dim=1))
        x7 = self.exp2(torch.cat((x3, x6), dim=1))
        x8 = self.exp3(torch.cat((x2, x7), dim=1))
        x9 = self.exp4(torch.cat((x1, x8), dim=1))

        y_hat = self.predict_proba(x9)

        return x9, y_hat


class UNet2(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        k, s, p = (3, 3), (1, 1), (1, 1)
        self.conv1a = nn.Conv2d(3, 64, k, s, p)
        self.bnorm1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, k, s, p)
        self.bnorm1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2a = nn.Conv2d(64, 128, k, s, p)
        self.bnorm2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, k, s, p)
        self.bnorm2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv3a = nn.Conv2d(128, 256, k, s, p)
        self.bnorm3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, k, s, p)
        self.bnorm3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv4a = nn.Conv2d(256, 512, k, s, p)
        self.bnorm4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, k, s, p)
        self.bnorm4b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv5a = nn.Conv2d(512, 1024, k, s, p)
        self.bnorm5a = nn.BatchNorm2d(1024)
        self.conv5b = nn.Conv2d(1024, 1024, k, s, p)
        self.bnorm5b = nn.BatchNorm2d(1024)
        self.up5 = nn.ConvTranspose2d(1024, 512, (2, 2), (2, 2), 0)
        self.bnorm5c = nn.BatchNorm2d(512)

        self.conv6a = nn.Conv2d(1024, 1024, k, s, p)
        self.bnorm6a = nn.BatchNorm2d(1024)
        self.conv6b = nn.Conv2d(1024, 512, k, s, p)
        self.bnorm6b = nn.BatchNorm2d(512)
        self.up6 = nn.ConvTranspose2d(512, 256, (2, 2), (2, 2), 0)
        self.bnorm6c = nn.BatchNorm2d(256)

        self.conv7a = nn.Conv2d(512, 512, k, s, p)
        self.bnorm7a = nn.BatchNorm2d(512)
        self.conv7b = nn.Conv2d(512, 256, k, s, p)
        self.bnorm7b = nn.BatchNorm2d(256)
        self.up7 = nn.ConvTranspose2d(256, 128, (2, 2), (2, 2), 0)
        self.bnorm7c = nn.BatchNorm2d(128)

        self.conv8a = nn.Conv2d(256, 256, k, s, p)
        self.bnorm8a = nn.BatchNorm2d(256)
        self.conv8b = nn.Conv2d(256, 128, k, s, p)
        self.bnorm8b = nn.BatchNorm2d(128)
        self.up8 = nn.ConvTranspose2d(128, 64, (2, 2), (2, 2), 0)
        self.bnorm8c = nn.BatchNorm2d(64)

        self.conv9a = nn.Conv2d(128, 128, k, s, p)
        self.bnorm9a = nn.BatchNorm2d(128)
        self.conv9b = nn.Conv2d(128, 64, k, s, p)
        self.bnorm9b = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(64, 1, (1, 1), (1, 1), 0)

    def forward(self, x):
        x1 = self.relu(self.bnorm1a(self.conv1a(x)))
        x1 = self.relu(self.bnorm1b(self.conv1b(x1)))
        x2 = self.pool1(x1)

        x2 = self.relu(self.bnorm2a(self.conv2a(x2)))
        x2 = self.relu(self.bnorm2b(self.conv2b(x2)))
        x3 = self.pool2(x2)

        x3 = self.relu(self.bnorm3a(self.conv3a(x3)))
        x3 = self.relu(self.bnorm3b(self.conv3b(x3)))
        x4 = self.pool3(x3)

        x4 = self.relu(self.bnorm4a(self.conv4a(x4)))
        x4 = self.relu(self.bnorm4b(self.conv4b(x4)))
        x5 = self.pool4(x4)

        x5 = self.relu(self.bnorm5a(self.conv5a(x5)))
        x5 = self.relu(self.bnorm5b(self.conv5b(x5)))
        x5 = self.up5(x5)
        x6 = torch.cat([x4, x5], dim=1)

        x6 = self.relu(self.bnorm6a(self.conv6a(x6)))
        x6 = self.relu(self.bnorm6b(self.conv6b(x6)))
        x6 = self.up6(x6)
        x7 = torch.cat([x3, x6], dim=1)

        x7 = self.relu(self.bnorm7a(self.conv7a(x7)))
        x7 = self.relu(self.bnorm7b(self.conv7b(x7)))
        x7 = self.up7(x7)
        x8 = torch.cat([x2, x7], dim=1)

        x8 = self.relu(self.bnorm8a(self.conv8a(x8)))
        x8 = self.relu(self.bnorm8b(self.conv8b(x8)))
        x8 = self.up8(x8)
        x9 = torch.cat([x1, x8], dim=1)

        x9 = self.relu(self.bnorm9a(self.conv9a(x9)))
        x9 = self.relu(self.bnorm9b(self.conv9b(x9)))

        x10 = self.conv10(x9)

        y_hat = self.sigm(x10)

        return y_hat
