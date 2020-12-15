import argparse
from model.VGG import *
from model.Decoder import *
from model.Transform import *
from model.GNN import *
import os
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import cv2 as cv
import numpy as np
import time
from dataloader.davis import DAVISLoader
from torch.utils import data

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content_dir', type=str, default='/home/lwq/sdb1/xiaoxin/data/DAVIS',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='style/style11.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='/home/lwq/sdb1/xiaoxin/code/SANeT_weight/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='/home/lwq/sdb1/xiaoxin/code/NST_GNN_result/experiment1/decoder_iter_100000.pth')
parser.add_argument('--transform', type=str, default='/home/lwq/sdb1/xiaoxin/code/NST_GNN_result/experiment1/transformer_iter_100000.pth')
parser.add_argument('--GNN', type=str, default='/home/lwq/sdb1/xiaoxin/code/NST_GNN_result/experiment1/GNN_iter_100000.pth')
parser.add_argument('--GNN2', type=str, default='/home/lwq/sdb1/xiaoxin/code/NST_GNN_result/experiment1/GNN2_iter_100000.pth')
# Additional options
parser.add_argument('--save_ext', default='output+',
                    help='The extension name of the output viedo')
parser.add_argument('--output', type=str, default='../output',
                    help='Directory to save the output image(s)')

# Advanced options
args = parser.parse_args('')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(args.output):
    os.mkdir(args.output)
decoder = Decoder('Decoder')
transform = Transform(in_planes=512)
vgg = VGG('VGG19')
GNN = CoattentionModel(all_channel=512)
GNN_2 = CoattentionModel(all_channel=512)

decoder.eval()
transform.eval()
vgg.eval()
GNN.eval()
GNN_2.eval()

# decoder.features.load_state_dict(torch.load(args.decoder))
decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.features.load_state_dict(torch.load(args.vgg))
GNN.load_state_dict(torch.load(args.GNN))
GNN_2.load_state_dict(torch.load(args.GNN2))

enc_1 = nn.Sequential(*list(vgg.features.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.features.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.features.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.features.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.features.children())[31:44])  # relu4_1 -> relu5_1


enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)
GNN.to(device)
GNN_2.to(device)


style_tf = test_transform()
style = style_tf(Image.open(args.style))
style = style.to(device).unsqueeze(0)

content_dataset = DAVISLoader(data_root=args.content_dir, num_sample=3, Training=False)
content_dataloader = data.DataLoader(content_dataset, batch_size=1)

seq = '0'
fps = 0
for i, batch in enumerate(content_dataloader):
    if batch['seq_name'] != seq:
        seq = batch['seq_name']
        fps = 0
    content_image1 = batch['content0'].to(device)
    content_image2 = batch['content1'].to(device)
    content_image3 = batch['content2'].to(device)

    with torch.no_grad():
        start = time.time()

        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)

        Content4_1_1 = enc_4(enc_3(enc_2(enc_1(content_image1))))
        Content4_1_2 = enc_4(enc_3(enc_2(enc_1(content_image2))))
        Content4_1_3 = enc_4(enc_3(enc_2(enc_1(content_image3))))

        Content5_1_1 = enc_5(Content4_1_1)
        Content5_1_2 = enc_5(Content4_1_2)
        Content5_1_3 = enc_5(Content4_1_3)
        Content4_1_1, _, _ = GNN(Content4_1_1, Content4_1_2, Content4_1_3)
        Content5_1_1, _, _ = GNN_2(Content5_1_1, Content5_1_2, Content5_1_3)
        Stylised = transform(Content4_1_1, Style4_1, Content5_1_1, Style5_1)

        content = decoder(Stylised)

        end = time.time()
        content.clamp(0, 255)
        content = content.cpu()
        content = content[0]
        content = content.transpose(1, 2)
        content = content.transpose(0, 2)
        content = content.numpy()*255

        output_value = np.clip(content, 0, 255).astype(np.uint8)
        output_value = cv.cvtColor(output_value, cv.COLOR_RGB2BGR)

        # print(seq)
        output_dir = os.path.join(args.output, seq[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_imagefile = os.path.join(output_dir, '%4d'%fps)+'.jpg'
        cv.imwrite(output_imagefile, output_value)
        print('save image:', output_imagefile)
        fps += 1