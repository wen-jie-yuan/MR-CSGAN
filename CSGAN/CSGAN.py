import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


##定义网络多尺度残差模块
class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)

        output = self.confusion(output)
        output = torch.add(output, identity_data)
        return output


##定义生成器
class Generator(torch.nn.Module):
    def __init__(self, num_channels, base_filter):
        super(Generator, self).__init__()
        ### 首先使用卷积层对原始图像进行采样压缩
        self.sample = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=32, stride=32, padding=0,
                      bias=False)
        )
        ### 然后初始化重建
        self.initialization = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels=base_filter, out_channels=1, kernel_size=32, stride=32, padding=0,
                               bias=False)
        )
        ## 体征提取，这一步出来的图像质量较好，但是细节部门还是缺失
        self.getFactor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.residual1 = self.make_layer(MSRB_Block)
        self.residual2 = self.make_layer(MSRB_Block)
        self.residual3 = self.make_layer(MSRB_Block)
        self.residual4 = self.make_layer(MSRB_Block)
        self.residual5 = self.make_layer(MSRB_Block)
        self.residual6 = self.make_layer(MSRB_Block)
        self.residual7 = self.make_layer(MSRB_Block)
        self.residual8 = self.make_layer(MSRB_Block)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.sample(x)
        outInitial = self.initialization(out)  ### 这一步初始化重建，重建出来的图像质量差
        out = self.getFactor(outInitial)
        LR = out
        out = self.residual1(out)
        concat1 = out
        out = self.residual2(out)
        concat2 = out
        out = self.residual3(out)
        concat3 = out
        out = self.residual4(out)
        concat4 = out
        out = self.residual5(out)
        concat5 = out
        out = self.residual6(out)
        concat6 = out
        out = self.residual7(out)
        concat7 = out
        out = self.residual8(out)
        concat8 = out
        out = torch.cat([LR, concat1, concat2, concat3, concat4, concat5, concat6, concat7, concat8], 1)
        out = self.out(out) + outInitial
        return out

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


##定义网络模型
def swish(x):
    # return x * torch.sigmoid(x)
    return torch.nn.LeakyReLU()(x)


##定义判决器
class Discriminator(nn.Module):
    def __init__(self, num_channel=1, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)
        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 2)
        self.conv4 = nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filter * 2)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.conv6 = nn.Conv2d(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(base_filter * 4)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filter * 8)
        self.conv8 = nn.Conv2d(base_filter * 8, base_filter * 8, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filter * 8)


        #FC layers
        self.fc1 = nn.Linear(base_filter * 8 * 16, 1024)
        self.fc2 = nn.Linear(1024, 1)

        #FCN
        self.conv9 = nn.Conv2d(base_filter * 8, num_channel, kernel_size=4, stride=1, padding=0)

    ##卷积表示
    def forward(self, x):
        x = swish(self.conv1(x))

        x = self.bn2(swish(self.conv2(x)))
        x = self.bn3(swish(self.conv3(x)))
        x = self.bn4(swish(self.conv4(x)))
        x = self.bn5(swish(self.conv5(x)))
        x = self.bn6(swish(self.conv6(x)))
        x = self.bn7(swish(self.conv7(x)))
        x = self.bn8(swish(self.conv8(x)))
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)



    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
