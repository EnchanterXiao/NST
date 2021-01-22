import torch
from torch.utils import data
import os
import cv2
import glob
import random
import numpy as np

from torchvision import transforms
from dataloader import custom_transforms as tr


class DAVISLoader(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, data_root, num_sample=3, Training=False):

        self.Training = Training
        self.augment_transform = None
        self._single_object = False
        self.num_sample = num_sample

        if Training:
            self.data_dir = os.path.join(data_root, 'Train')
            self.augment_transform = transforms.Compose([
                tr.Resize(scales=(512, 512)),
                tr.Crop(scale=0.5),
                tr.ToTensor()
                ])
        else:
            self.data_dir = os.path.join(data_root, 'Test')
            self.augment_transform = transforms.Compose([
                tr.Resize(scales=(512, 512)),
                tr.ToTensor()
            ])

        # Load Videos
        self.videos = []
        for seq in sorted(os.listdir(self.data_dir)):
            self.videos.append(seq)

        if Training:
            random.shuffle(self.videos)

        self.videoindex = {}
        self.imagefiles = []
        self.videofiles = []
        offset = 0
        for _video in self.videos:
            imagefiles = sorted(glob.glob(os.path.join(self.data_dir, _video, '*.jpg')))

            self.imagefiles.extend(imagefiles)
            self.videofiles.extend([_video] * len(imagefiles))
            self.videoindex[_video] = [offset, offset + len(imagefiles)]
            offset += len(imagefiles)
        print('total video: ', len(self.videos))
        print('total image: ', len(self.imagefiles))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, index):
        samples = {}

        video = self.videofiles[index]
        video_index = self.videoindex[video]
        imagefile1 = self.imagefiles[index]
        image = cv2.imread(imagefile1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        samples['content{}'.format(0)] = image

        index2 = min(index + 1, video_index[1] - 1)
        imagefile2 = self.imagefiles[index2]
        image2 = cv2.imread(imagefile2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        samples['content{}'.format(1)] = image2

        if self.augment_transform is not None:
            samples = self.augment_transform(samples)

        samples['seq_name'] = video
        return samples



if __name__ == '__main__':
    data = DAVISLoader(data_root='/home/lwq/sdb1/xiaoxin/data/YoutubeVOS', num_sample=3, Training=True)
    # data = DAVISLoader(data_root='/home/lwq/sdb1/xiaoxin/data/DAVIS', num_sample=3, Training=False)
    samples = data.__getitem__(0)
    for i in samples:
        print(i)
    print(samples['content0'])


