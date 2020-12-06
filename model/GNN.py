import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model import ConvGRU


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class GNN(nn.Module):
    def __init__(self, all_channel=512):
        super(GNN, self).__init__()
        #self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        # self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        # self.gate_s = nn.Sigmoid()

        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.conv_fusion = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=True)
        self.conv_fusion_output = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=True)
        self.propagate_layers = 5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, input1, input2, input3):  # 注意input2 可以是多帧图像
        # print(input1.shape)
        batch_num = input1.size()[0]
        x1s = torch.zeros_like(input1).cuda()
        x2s = torch.zeros_like(input2).cuda()
        x3s = torch.zeros_like(input3).cuda()
        for ii in range(batch_num):
            exemplar = input1[ii, :, :, :][None].contiguous().clone()
            query = input2[ii, :, :, :][None].contiguous().clone()
            query1 = input3[ii, :, :, :][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):

                attention1 = self.conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                                         self.generate_attention(exemplar, query1)],
                                                        1))
                attention2 = self.conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                                         self.generate_attention(query, query1)], 1))
                attention3 = self.conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                                         self.generate_attention(query1, query)], 1))

                h_v1 = self.ConvGRU(attention1, exemplar)
                h_v2 = self.ConvGRU(attention2, query)
                h_v3 = self.ConvGRU(attention3, query1)
                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()

                if passing_round == self.propagate_layers - 1:
                    x1s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v1, input1[ii, :, :, :][None].contiguous()], 1))
                    x2s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v2, input2[ii, :, :, :][None].contiguous()], 1))
                    x3s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v3, input3[ii, :, :, :][None].contiguous()], 1))

        return x1s, x2s, x3s

    def generate_attention(self, exemplar, query):  #h*w*c
        fea_size = query.size()[2:]
        # print(query.shape)
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  #N,HW,C
        # exemplar_corr = self.linear_e(exemplar_t)
        exemplar_corr = exemplar_t
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        # input1_mask = self.gate(input1_att)
        # input1_mask = self.gate_s(input1_mask)
        # input1_att = input1_att * input1_mask

        return input1_att


class CoattentionModel(nn.Module):
    def __init__(self, all_channel=512):
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.conv_fusion = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=True)
        self.propagate_layers = 3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2, input3):  # 注意input2 可以是多帧图像
        # print(input1.shape)
        batch_num = input1.size()[0]
        x1s = torch.zeros_like(input1).cuda()
        x2s = torch.zeros_like(input2).cuda()
        x3s = torch.zeros_like(input3).cuda()
        for ii in range(batch_num):
            exemplar = input1[ii, :, :, :][None].contiguous().clone()
            query = input2[ii, :, :, :][None].contiguous().clone()
            query1 = input3[ii, :, :, :][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):

                attention1 = self.conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                                         self.generate_attention(exemplar, query1)],
                                                        1))
                attention2 = self.conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                                         self.generate_attention(query, query1)], 1))
                attention3 = self.conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                                         self.generate_attention(query1, query)], 1))

                h_v1 = self.ConvGRU(attention1, exemplar)
                h_v2 = self.ConvGRU(attention2, query)
                h_v3 = self.ConvGRU(attention3, query1)
                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()

                if passing_round == self.propagate_layers - 1:
                    x1s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v1, input1[ii, :, :, :][None].contiguous()], 1))
                    x2s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v2, input2[ii, :, :, :][None].contiguous()], 1))
                    x3s[ii, :, :, :] = self.conv_fusion_output(torch.cat([h_v3, input3[ii, :, :, :][None].contiguous()], 1))

        return x1s, x2s, x3s

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input1_mask = self.gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        return input1_att
