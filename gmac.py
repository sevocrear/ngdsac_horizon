import torch
import numpy as np

from torchvision import transforms

from skimage import color
from skimage.io import imsave
from skimage.draw import line, set_color, disk

from model import Model

import time
import warnings
import argparse
import os

from ngdsac import NGDSAC
from loss import Loss
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='Estimate horizon lines using a trained (NG-)DSAC network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', '-m', default='models/weights_ngdsac_pretrained.net', 
	help='a trained network')

parser.add_argument('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of the network, must match the model to load (multiplicative factor for number of channels)')

parser.add_argument('--imagesize', '-is', type=int, default=256, 
	help='size of input images to the network, must match the model to load')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count, relative to image size')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.1, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--hypotheses', '-hyps', type=int, default=16, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely, use with a DSAC model')

parser.add_argument('--scorethreshold', '-st', type=float, default=0.4, 
	help='threshold on soft inlier count for drawing the estimate (range 0-1)')
parser.add_argument('--device', '-d', type=str, default="cuda", 
	help='Device to run the neural network (cuda, cpu)')

opt = parser.parse_args()

# setup ng dsac estimator
ngdsac = NGDSAC(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, Loss(opt.imagesize), 1)

device = opt.device
# load network
nn = Model(opt.capacity)
nn.load_state_dict(torch.load(opt.model, map_location=torch.device(device)))
nn.eval()

nn = nn.to(device)

macs, params = get_model_complexity_info(nn, (3, opt.imagesize, opt.imagesize), as_strings=False,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs/1e9))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
