import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torch.utils.model_zoo as model_zoo

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(3, 3, (1, 1))]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class VGG19(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG19, self).__init__()
        vgg19 = VGG('VGG19')
        if init_weights is False:
            vgg19.features.load_state_dict(torch.load('/home/lwq/sdb1/xiaoxin/code/SANeT_weight/vgg_normalised.pth'))

        model_list = list(vgg19.features.children())
        # for i in range(len(model_list)):
        #     print(i, model_list[i])
        self.conv0_1 = nn.Sequential(model_list[0])
        self.conv1_1 = nn.Sequential(*model_list[1:4])
        self.conv1_2 = nn.Sequential(*model_list[4:7])
        self.maxpool_1 = nn.Sequential(model_list[7])
        self.conv2_1 = nn.Sequential(*model_list[8:11])
        self.conv2_2 = nn.Sequential(*model_list[11:14])
        self.maxpool_2 = nn.Sequential(model_list[14])
        self.conv3_1 = nn.Sequential(*model_list[15:18])
        self.conv3_2 = nn.Sequential(*model_list[18:21])
        self.conv3_3 = nn.Sequential(*model_list[21:24])
        self.conv3_4 = nn.Sequential(*model_list[24:27])
        self.maxpool_3 = nn.Sequential(model_list[27])
        self.conv4_1 = nn.Sequential(*model_list[28:31])
        self.conv4_2 = nn.Sequential(*model_list[31:34])
        self.conv4_3 = nn.Sequential(*model_list[34:37])
        self.conv4_4 = nn.Sequential(*model_list[37:40])
        self.maxpool_4 = nn.Sequential(model_list[40])
        self.conv5_1 = nn.Sequential(*model_list[41:44])
        self.conv5_2 = nn.Sequential(*model_list[44:47])
        self.conv5_3 = nn.Sequential(*model_list[47:50])
        self.conv5_4 = nn.Sequential(*model_list[50:53])
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = {}
        x = self.conv0_1(x)
        out['conv1_1'] = self.conv1_1(x)
        out['conv1_2'] = self.conv1_2(out['conv1_1'])
        out['pool1'] = self.maxpool_1(out['conv1_2'])

        out['conv2_1'] = self.conv2_1(out['pool1'])
        out['conv2_2'] =self.conv2_2(out['conv2_1'])
        out['pool2'] = self.maxpool_2(out['conv2_2'])

        out['conv3_1'] = self.conv3_1(out['pool2'])
        out['conv3_2'] = self.conv3_2(out['conv3_1'])
        out['conv3_3'] = self.conv3_3(out['conv3_2'])
        out['conv3_4'] = self.conv3_4(out['conv3_3'])
        out['pool3'] = self.maxpool_3(out['conv3_4'])

        out['conv4_1'] = self.conv4_1(out['pool3'])
        out['conv4_2'] = self.conv4_2(out['conv4_1'])
        out['conv4_3'] = self.conv4_3(out['conv4_2'])
        out['conv4_4'] = self.conv4_4(out['conv4_3'])
        out['pool4'] = self.maxpool_4(out['conv4_4'])
        # print(out['pool4'].shape)

        out['conv5_1'] = self.conv5_1(out['pool4'])
        out['conv5_2'] = self.conv5_2(out['conv5_1'])
        out['conv5_3'] = self.conv5_3(out['conv5_2'])
        out['conv5_4'] = self.conv5_4(out['conv5_3'])
        # out['pool5'] = F.max_pool2d(out['conv5_4'], kernel_size=2, stride=2)

        # return [out[key] for key in out_key]
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == "__main__":
    net = VGG19()
    print(net)