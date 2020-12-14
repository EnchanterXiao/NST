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


def conv_block(name, in_C, out_C, activation='ReLU', kernel_size=3, stride=1, padding=1, transpose=False):
    block = nn.Sequential()

    if not transpose:
        func = nn.Conv2d
        conv = ' Conv'
    else:
        func = nn.ConvTranspose2d
        conv = ' DeConv'
    if activation == 'ReLU':
        activate = nn.ReLU
    elif activation == 'Tanh':
        activate = nn.Tanh

    block.add_module(name + conv, func(in_C, out_C, kernel_size, stride, padding))
    block.add_module(name + ' Inst_norm', nn.InstanceNorm2d(out_C))
    if activation == 'ReLU':
        block.add_module(name + ' ' + activation, activate(inplace=True))
    elif activation == 'Tanh':
        block.add_module(name + ' ' + activation, activate())

    return block

class NSTNet(nn.Module):
    def __init__(self, start_iter):
        super(NSTNet, self).__init__()
        self.content_encoder = VGG('VGG19')
        self.con_enc_1 = nn.Sequential(*list(self.content_encoder.children())[:4])  # input -> relu1_1
        self.con_enc_2 = nn.Sequential(*list(self.content_encoder.children())[4:11])  # relu1_1 -> relu2_1
        self.con_enc_3 = nn.Sequential(*list(self.content_encoder.children())[11:18])  # relu2_1 -> relu3_1
        self.con_enc_4 = nn.Sequential(*list(self.content_encoder.children())[18:31])  # relu3_1 -> relu4_1
        self.con_enc_5 = nn.Sequential(*list(self.content_encoder.children())[31:44])  # relu4_1 -> relu5_1

        self.style_encoder = VGG('VGG19')
        self.sty_enc_1 = nn.Sequential(*list(self.style_encoder.children())[:4])  # input -> relu1_1
        self.sty_enc_2 = nn.Sequential(*list(self.style_encoder.children())[4:11])  # relu1_1 -> relu2_1
        self.sty_enc_3 = nn.Sequential(*list(self.style_encoder.children())[11:18])  # relu2_1 -> relu3_1
        self.sty_enc_4 = nn.Sequential(*list(self.style_encoder.children())[18:31])  # relu3_1 -> relu4_1
        self.sty_enc_5 = nn.Sequential(*list(self.style_encoder.children())[31:44])  # relu4_1 -> relu5_1

        self.transform = Transform(in_planes=512)
        self.decoder = Decoder('Decoder_v2')
        if (start_iter > 0):
            self.transform.load_state_dict(torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('/home/lwq/sdb1/xiaoxin/code/SANT_weight/decoder_iter_' + str(start_iter) + '.pth'))

    def extract_content_fea(self, content_img):
        pass

    def extract_style_fea(self, style_img):
        pass

    def forward(self, content_img, style_img):
        content_feas = self.extract_content_fea(content_img)
        content_feas_s = self.extract_content_fea(style_img)
        style_feas = self.extract_style_fea(style_img)

        stylized_fea = self.transform(content_feas[3],content_feas_s[3], style_feas[3],
                                      content_feas[4],content_feas_s[4], style_feas[4])
        stylized_img = self.decoder(stylized_fea)
        return stylized_img

    def compute_loss(self, content_img, style_img):
        pass


if __name__ == '__main__':
    style_net = NSTNet()

    one = Variable(torch.ones(1, 3, 436, 436))
    res = style_net(one)
    print(style_net)