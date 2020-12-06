import numpy as np
import argparse
import os
import torch
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from model.Decoder import *
from model.VGG import *
from model.SANet import *
from model.model import *
from dataloader.dataset import *
from dataloader.davis import *

import numpy as np
from torch.utils import data
import torchvision.models as models


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] / (1.0 + args.lr_decay * iteration_count)
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/home/lwq/sdb1/xiaoxin/data/coco_2014/data/coco_2014/images/train2014/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/home/lwq/sdb1/xiaoxin/data/wikiArt/',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='../SANeT_weight/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='../NST_GNN_result/experiment1',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='../NST_GNN_result/log1',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')
#args.content_dir = '/home/lwq/sdb1/xiaoxin/data/DAVIS'
args.content_dir = '/home/lwq/sdb1/xiaoxin/data/YoutubeVOS'

device = torch.device('cuda')

decoder = Decoder('Decoder')

vgg = VGG('VGG19')
vgg.features.load_state_dict(torch.load(args.vgg))
# vgg = models.vgg19(pretrained=True)
vgg = nn.Sequential(*list(vgg.features.children())[:44])
network = Net(vgg, decoder, args.start_iter)

network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

# content_dataset = FlatFolderDataset(args.content_dir, content_tf)
content_dataset = DAVISLoader(data_root=args.content_dir, num_sample=3, Training=True)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([
                              {'params': network.decoder.parameters(), 'lr':1*args.lr},
                              {'params': network.transform.parameters(), 'lr':1 *args.lr},
                              {'params': network.GNN.parameters(), 'lr':1*args.lr}],
                              lr=args.lr)

if(args.start_iter > 0):
    if os.path.exists('optimizer_iter_' + str(args.start_iter) + '.pth'):
        optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

writer = SummaryWriter('/home/lwq/sdb1/xiaoxin/code/NST_GNN_result/runs/loss')

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter)
    if i%3==0:
        content_image1 = content_images['content0'].to(device)
        content_image2 = content_images['content0'].to(device)
        content_image3 = content_images['content0'].to(device)
    else:
        content_image1 = content_images['content0'].to(device)
        content_image2 = content_images['content1'].to(device)
        content_image3 = content_images['content2'].to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network(content_image1,
                            content_image2,
                            content_image3,
                            style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s
    writer.add_scalar('total loss', loss, global_step=i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.GNN.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/GNN_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.GNN_2.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/GNN2_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

writer.close()