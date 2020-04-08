import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
from math import ceil
import os.path as osp
import shutil
import tempfile

import mmcv
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from pytorchcv.model_provider import get_model as ptcv_get_model


def own_loss(A, B):
    return (A - B).norm()**2 / A.size(0)


class output_hook(object):
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        tmp_tensor = output
        if self.outputs is None:
            self.outputs = tmp_tensor
        else:
            self.outputs = torch.cat((self.outputs, tmp_tensor), dim=0)

    def clear(self):
        self.outputs = None


def getDistillData(teacher_model, dataloader, work_dir, num_batch):
    file_name = f'retinanet_distill.pth'

    print('generating ditilled data')

    hooks, hook_handles, bn_stats, distill_data = [], [], [], []

    for n, m in teacher_model.backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            eps = 1e-6
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))

    assert len(hooks) == len(bn_stats)
    teacher_model = MMDataParallel(teacher_model, device_ids=[0])
    teacher_model.eval()

    logs = []
    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        logs.append({})
        # Uniform initilizaition and normalize to 0
        size = (1, 3, 800, 1216)
        gaussian_data['img'] = [
            (torch.randint(high=255, size=size) - 128).float().cuda() / 5418.75
        ]

        gaussian_data['img'][0].requires_grad = True
        optimizer = optim.Adam([gaussian_data['img'][0]], lr=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-4, verbose=False, patience=100)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(10):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(return_loss=False,
                                   rescale=True,
                                   **gaussian_data)
            mean_loss = 0
            std_loss = 0
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
                # print(cnt, mean_loss.item(), std_loss.item())
            tmp_mean = torch.mean(gaussian_data['img'][0].view(size[0], 3, -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(gaussian_data['img'][0].view(size[0], 3, -1), dim=2)
                + eps)
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            total_loss = mean_loss + std_loss
            total_loss.backward()
            # print(it, mean_loss.item(), std_loss.item())
            logs[-1][it] = copy.deepcopy(total_loss.item())
            optimizer.step()
            # scheduler.step(total_loss.item())
            # if total_loss <= (len(hooks) + 1) * 10:
            # 	break
        gaussian_data['img'][0] = gaussian_data['img'][0].detach().clone()
        distill_data.append(gaussian_data)

    for handle in hook_handles:
        handle.remove()

    torch.save(distill_data, file_name)
    json.dump(logs, open(f'loss_log.json', 'w'))

    return distill_data


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   imgs_per_gpu=1,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    getDistillData(teacher_model=model,
                   dataloader=data_loader,
                   work_dir='',
                   num_batch=1)


if __name__ == '__main__':
    main()
