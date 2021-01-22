import numpy as np
import argparse
import os
import torch
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from model_v2.Decoder import *
from model_v2.model_v2 import *
from dataloader.dataset import *
from dataloader.video_dataset import *

import numpy as np
from torch.utils import data

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/home/lwq/sdb1/xiaoxin/data/coco_2014/data/coco_2014/images/train2014/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/home/lwq/sdb1/xiaoxin/data/wikiArt/',
                    help='Directory path to a batch of style images')
# training options
parser.add_argument('--save_dir', default='../NST_v2/experiments3',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='../NST_v2/logs3',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=600000)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--start_iter', type=float, default=500000)
args = parser.parse_args('')
#args.content_dir = '/home/lwq/sdb1/xiaoxin/data/DAVIS'
args.content_dir = '/home/lwq/sdb1/xiaoxin/data/YoutubeVOS'

device = torch.device('cuda')

network = NSTNet(args.start_iter)

network.train()
network.to(device)

style_tf = train_transform()

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
                              {'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()},
                              {'params': network.content_encoder.parameters()},
                              {'params': network.style_encoder.parameters()}], lr=args.lr)

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('../NST_v2/experiments/optimizer_iter_' + str(args.start_iter) + '.pth'))

writer = SummaryWriter(os.path.join(args.save_dir, 'runs/loss'))

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter)
    content_img0 = content_images['content0'].to(device)
    content_img1 = content_images['content1'].to(device)
    style_images = next(style_iter).to(device)
    # print(content_images.shape)
    # print(style_images.shape)
    loss_c, loss_s, l_identity1, l_identity2, loss_t = network.forward_video(content_img0, content_img1, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1 + loss_t*10
    writer.add_scalar('total loss', loss, global_step=i)
    writer.add_scalar('content loss', loss_c, global_step=i)
    writer.add_scalar('style loss', loss_s, global_step=i)
    writer.add_scalar('identity1 loss', l_identity1*50, global_step=i)
    writer.add_scalar('identity2 loss', l_identity2, global_step=i)
    writer.add_scalar('temporal loss', loss_t*100, global_step=i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
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
        state_dict = network.content_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/content_encoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.style_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/style_encoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

writer.close()