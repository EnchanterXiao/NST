import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from model.VGG import VGG
from model.Decoder import Decoder
from model.Transform import *
from model.SANet import *
from model.GNN import GNN,CoattentionModel


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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


class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        vgg = encoder
        # self.enc_0 = nn.Sequential(*list(vgg.children())[:1])
        # enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1
        # transform
        self.transform = Transform(in_planes=512)
        self.GNN = CoattentionModel(all_channel=512)
        self.GNN_2 = CoattentionModel(all_channel=512)
        self.decoder = decoder
        if (start_iter > 0):
            self.transform.load_state_dict(torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        self.variation_loss = nn.L1Loss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.dx_bias = np.zeros([256, 256])
        self.dy_bias = np.zeros([256, 256])
        for i in range(256):
            self.dx_bias[:, i] = i
            self.dx_bias[i, :] = i

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]

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

    def calc_temporal_loss(self,  x1, x2):
        h = x1.shape[2]
        w = x1.shape[3]
        D = h*w
        return self.mse_loss(x1, x2)

    def compute_total_variation_loss_l1(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        h1 = inputs[:, :, 0:h-1, :]
        h2 = inputs[:, :, 1:h, :]
        w1 = inputs[:, :, :, 0:w-1]
        w2 = inputs[:, :, :, 1:w]
        return self.variation_loss(h1, h2)+self.variation_loss(w1, w2)

    def forward(self, content1, content2, content3, style):
        # feature extract
        style_feats = self.encode_with_intermediate(style)
        content1_feats = self.encode_with_intermediate(content1)
        content2_feats = self.encode_with_intermediate(content2)
        content3_feats = self.encode_with_intermediate(content3)
        gcontent1_feats, gcontent2_feats, gcontent3_feats = self.GNN(content1_feats[3], content2_feats[3],
                                                                          content3_feats[3])

        ggcontent1_feats, ggcontent2_feats, ggcontent3_feats = self.GNN_2(content1_feats[4], content2_feats[4],
                                                                           content3_feats[4])

        # feature fusion & propagation
        stylized_1 = self.transform(gcontent1_feats, style_feats[3], ggcontent1_feats, style_feats[4])
        stylized_2 = self.transform(gcontent2_feats, style_feats[3], ggcontent2_feats, style_feats[4])
        stylized_3 = self.transform(gcontent3_feats, style_feats[3], ggcontent3_feats, style_feats[4])

        stylized = torch.cat((stylized_1, stylized_2, stylized_3), 0)
        content_feats_l3 = torch.cat((content1_feats[3], content2_feats[3], content3_feats[3]), 0)
        content_feats_l4 = torch.cat((content1_feats[4], content2_feats[4], content3_feats[4]), 0)
        # decoder
        g_t = self.decoder(stylized)

        # compute loss
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats_l3, norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats_l4, norm=True)

        style_feats[0] = torch.cat((style_feats[0], style_feats[0], style_feats[0]), 0)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            style_feats[i] = torch.cat((style_feats[i], style_feats[i], style_feats[i]), 0)
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return loss_c, loss_s