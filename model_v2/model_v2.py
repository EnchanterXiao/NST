'''
encoder不用预训练初始化，用一个比VGG19小一点的网络，预训练的VGG19作损失网络
'''
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model_v2.VGG import *
from model_v2.Decoder import *
from model_v2.Transform import *

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


class NSTNet(nn.Module):
    def __init__(self, start_iter):
        super(NSTNet, self).__init__()
        self.content_encoder = VGG19
        self.style_encoder = VGG19
        self.transform = Transform(in_planes=512)
        self.decoder = Decoder('Decoder_v2')

        self.loss_net = VGG19
        state_dict = torch.load('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth')
        state_dict = {k: v for k, v in state_dict.items() if 'class' not in k}
        self.loss_net.load_state_dict(state_dict)

        if (start_iter > 0):
            self.transform.load_state_dict(
                torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(
                torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/decoder_iter_' + str(start_iter) + '.pth'))

        self.mse_loss = nn.MSELoss()



    def forward(self, content_img, style_img):
        content_feas = self.content_encoder(content_img)
        content_feas_s = self.content_encoder(style_img)
        style_feas = self.style_encoder(style_img)
        stylized_fea = self.transform(content_feas['conv4_1'], content_feas_s['conv4_1'], style_feas['conv4_1'],
                                      content_feas['conv5_1'], content_feas_s['conv5_1'], style_feas['conv5_1'])
        stylized_img = self.decoder(stylized_fea)

        #计算loss
        loss = 0
        return loss

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_temporal_loss(self, x1, x2):
        h = x1.shape[2]
        w = x1.shape[3]
        D = h * w
        return self.mse_loss(x1, x2)


if __name__ == '__main__':
    style_net = NSTNet()

    one = Variable(torch.ones(1, 3, 436, 436))
    res = style_net(one)
    print(style_net)