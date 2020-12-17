import torch
import torch.nn as nn
import torch.nn.functional as F
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG19(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG19, self).__init__()
        features = self._make_layers(cfg['VGG19'], batch_norm=True)
        model_list = list(features.children())
        # for i in range(len(model_list)):
        #     print(i, model_list[i])
        self.conv1_1 = model_list[0]
        self.conv1_2 = model_list[3]
        self.conv2_1 = model_list[7]
        self.conv2_2 = model_list[10]
        self.conv3_1 = model_list[14]
        self.conv3_2 = model_list[17]
        self.conv3_3 = model_list[20]
        self.conv3_4 = model_list[23]
        self.conv4_1 = model_list[27]
        self.conv4_2 = model_list[30]
        self.conv4_3 = model_list[33]
        self.conv4_4 = model_list[36]
        self.conv5_1 = model_list[40]
        self.conv5_2 = model_list[43]
        self.conv5_3 = model_list[46]
        self.conv5_4 = model_list[49]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = {}
        out['conv1_1'] = F.relu(self.conv1_1(x))
        out['conv1_2'] = F.relu(self.conv1_2(out['conv1_1']))
        out['pool1'] = F.max_pool2d(out['conv1_2'], kernel_size=2, stride=2)

        out['conv2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['conv2_2'] = F.relu(self.conv2_2(out['conv2_1']))
        out['pool2'] = F.max_pool2d(out['conv2_2'], kernel_size=2, stride=2)

        out['conv3_1'] = F.relu(self.conv3_1(out['pool2']))
        out['conv3_2'] = F.relu(self.conv3_2(out['conv3_1']))
        out['conv3_3'] = F.relu(self.conv3_3(out['conv3_2']))
        out['conv3_4'] = F.relu(self.conv3_4(out['conv3_3']))
        out['pool3'] = F.max_pool2d(out['conv3_4'], kernel_size=2, stride=2)

        out['conv4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['conv4_2'] = F.relu(self.conv4_2(out['conv4_1']))
        out['conv4_3'] = F.relu(self.conv4_3(out['conv4_2']))
        out['conv4_4'] = F.relu(self.conv4_4(out['conv4_3']))
        out['pool4'] = F.max_pool2d(out['conv4_4'], kernel_size=2, stride=2)
        # print(out['pool4'].shape)

        out['conv5_1'] = F.relu(self.conv5_1(out['pool4']))
        out['conv5_2'] = F.relu(self.conv5_2(out['conv5_1']))
        out['conv5_3'] = F.relu(self.conv5_3(out['conv5_2']))
        out['conv5_4'] = F.relu(self.conv5_4(out['conv5_3']))
        out['pool5'] = F.max_pool2d(out['conv5_4'], kernel_size=2, stride=2)

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

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG19()
    print(net)