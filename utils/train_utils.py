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

import torch
import os
import torch.nn as nn
from progress.bar import Bar


def test(model, test_loader):
    """
    test a model on a given dataset
    """
    total, correct = 0, 0
    bar = Bar('Testing', max=len(test_loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total

            bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
            bar.next()
    print('\nFinal acc: %.2f%% (%d/%d)' % (100. * acc, correct, total))
    bar.finish()
    model.train()
    return acc


def update(quantized_model, distilD):
    """
    Update activation range according to distilled data
    quantized_model: a quantized model whose activation range to be updated 
    distilD: distilled data
    """
    with torch.no_grad():
        for batch_idx, inputs in enumerate(distilD):
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.cuda()
            outputs = quantized_model(inputs)
    return quantized_model
