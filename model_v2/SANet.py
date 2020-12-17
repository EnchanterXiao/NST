import torch
import torch.nn as nn

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

class SANet(nn.Module):

    def __init__(self):
        super(SANet, self).__init__()
        name = 'SANet'
        self.sm = nn.Softmax(dim=-1)

    def forward(self, content, content_s, style):
        b, c, h, w = content.size()
        Fc = content.view(b, -1, w*h).permute(0, 2, 1)
        Fs = content_s.view(b, -1, w*h)
        S = torch.bmm(Fc, Fs)
        S = self.sm(S)

        style = style.view(b, -1, w*h)
        O = torch.bmm(style, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        output = torch.cat((content, O), dim=1)
        return output