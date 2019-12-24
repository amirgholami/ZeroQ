#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='zero order analysis for quantized model')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=512,
                        help='batch size of test data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print('******full precision model loaded******')

    # load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet/',
                              for_inception=args.model.startswith('inception'))
    # generate distilled data
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    print('******data loaded******')

    # quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model)
    # freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # update activation range according to distilled data
    update(quantized_model, dataloader)
    # freeze activation range during test
    freeze_model(quantized_model)
    quantized_model = nn.DataParallel(quantized_model).cuda()
    test(quantized_model, test_loader)
