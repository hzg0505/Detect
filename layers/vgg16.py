# --coding:utf-8--
import torch
import torch.nn as nn
import torchvision.models as models

# 定义主干网络
class MY_SSD_VGG(nn.Module):
    def __init__(self, pretrained=True, WH=300):
        super(MY_SSD_VGG, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        if WH == 300:
            features[23] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # 500
        # else:
        #     features[23] = nn.MaxPool2d(kernel_size=2, stride=2)
        # maxpool2d: 6, 13 23 33 43
        self.layer1 = nn.Sequential(*features[0:33])  # w,h/8  conv4_3
        # 300

        self.layers = features[33:43]  # conv5_3
        self.layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)]
        self.layers += [nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)]
        self.layers += [nn.Conv2d(1024, 1024, kernel_size=1)]
        self.layer2 = nn.Sequential(*self.layers)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        # self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        f_list = []
        x = self.layer1(x)
        f_list.append(x)
        x = self.layer2(x)
        # x = self.maxpool5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        f_list.append(x)
        # print(out1.shape, out2.shape)
        return f_list




        # 区分两种输入 300*300  500*500
        # 300*300时在 第三个maxpooling时 ceil_mode=True

# 定义额外的卷积网络
class EXTRA_CONV(nn.Module):
    def __init__(self):
        super(EXTRA_CONV, self).__init__()
        self.extra_layer1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 10*10
        )
        self.extra_layer2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 5*5
        )
        self.extra_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 3*3
        )
        self.extra_layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # 1*1
        )

    def forward(self, x):
        f_list = []
        x = self.extra_layer1(x)
        f_list.append(x)
        x = self.extra_layer2(x)
        f_list.append(x)
        x = self.extra_layer3(x)
        f_list.append(x)
        x = self.extra_layer4(x)
        f_list.append(x)
        return f_list, x

# 定义SSD模型
class SSD(nn.Module):
    def __init__(self, phase, size, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.size = size
        self.base = MY_SSD_VGG(pretrained=True, WH=size)
        self.extras = EXTRA_CONV()
        self.head = head
        self.num_classes = num_classes

    def cls_predictor(num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                         kernel_size=3, padding=1)

    def bbox_predictor(num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        loc, conf = [], []
        features = []
        features = self.base(x)
        x = features[1]
        f2, x = self.extras(x)
        features = features + f2

        # for feature in enumerate(features):
        #     feature =

        return features



# net = MY_SSD_VGG(WH=500)
# print(net)
# x = torch.randn(1, 3, 300, 300)
# out1, out2 = net(x)
# print(out1.shape, out2.shape)
vgg = MY_SSD_VGG(WH=500)
extra = EXTRA_CONV()
net = SSD("Train", 300, "N", num_classes=10)
x = torch.randn(1, 3, 300, 300)
feature = net(x)

print(net)

# # print(feature[1].shape)
# for i in range(len(feature)):
#     print(feature[i].shape)