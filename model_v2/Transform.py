from .SANet import *

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet = SANet()
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes*2, in_planes, (3, 3))
    def forward(self, content4_1, content4_1_s, style4_1, content5_1, content5_1_s, style5_1):
        #print(content4_1.shape)
        return self.merge_conv(self.merge_conv_pad(self.sanet(content4_1, content4_1_s, style4_1) + self.upsample5_1(self.sanet(content5_1, content5_1_s, style5_1))))