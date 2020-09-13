import random
import cv2
import numpy as np
import torch
import math
from torch.autograd import Variable


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, scales=(512, 512)):
        self.scales = scales

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = sample[elem]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, self.scales, interpolation=flagval)

            sample[elem] = tmp

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'fname' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample


class Crop(object):
    def __init__(self, scale=0.9):
        self.scale = scale

    def __call__(self, sample):
        for elem in sample.keys():
            H = int(self.scale * sample[elem].shape[0])
            W = int(self.scale * sample[elem].shape[1])
            H_offset = random.choice(range(sample[elem].shape[0] - H))
            W_offset = random.choice(range(sample[elem].shape[1] - W))
            H_slice = slice(H_offset, H_offset + H)
            W_slice = slice(W_offset, W_offset + W)
            break
        for elem in sample.keys():
            if sample[elem].ndim == 2:
                sample[elem] = sample[elem][H_slice, W_slice]
            else:
                sample[elem] = sample[elem][H_slice, W_slice, :]
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            tmp = tmp / 255
            sample[elem] = torch.from_numpy(tmp).type(torch.FloatTensor)

        return sample

