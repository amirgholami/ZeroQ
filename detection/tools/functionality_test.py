import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from portable_quantizer import *

parser = argparse.ArgumentParser(
    description='functionality test of portable quantizer')
parser.add_argument('--dataset',
                    type=str,
                    default='imagenet',
                    metavar='D',
                    help='dataset')
parser.add_argument('--model',
                    type=str,
                    default='resnet18',
                    help='model name used in pytorchcv')
parser.add_argument('--train',
                    action='store_true',
                    help='train or test the model')
parser.add_argument('--update',
                    action='store_true',
                    help='update the bn stats and quantization range')
parser.add_argument('--epoch', type=int, default=10, help='training epoch')
parser.add_argument('--num_data',
                    type=int,
                    default=None,
                    metavar='N',
                    help='number of batches')
parser.add_argument('--batch_size',
                    type=int,
                    default=1024,
                    metavar='B',
                    help='size of batches')
parser.add_argument('--conv_bit',
                    type=int,
                    default=8,
                    metavar='C',
                    help='bit number of convolution layer')
parser.add_argument('--act_bit',
                    type=int,
                    default=8,
                    metavar='A',
                    help='bit number of activation layer')
parser.add_argument('--wc', action='store_true', help='weight channelwise')
parser.add_argument('--wp', action='store_true', help='weight percentile')
parser.add_argument('--ap', action='store_true', help='activation percentile')
parser.add_argument('--quantize', action='store_true', help='quantize model')
parser.add_argument('--work_dir',
                    type=str,
                    default=None,
                    help='path to save checkpoint')
parser.add_argument('--resume_from',
                    type=str,
                    default=None,
                    help='path to load checkpoint')
parser.add_argument('--quant_mode',
                    type=str,
                    default='asymmetric',
                    choices=['asymmetric', 'symmetric'],
                    help='quantize mode used')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

args = parser.parse_args()

torch.manual_seed(17)
np.random.seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not train:
    test_loader = getTestData(args.dataset,
                              num_data=args.num_data,
                              batch_size=args.batch_size,
                              path='/rscratch/data/imagenet12/')
else:
    train_loader, test_loader = getData(args.dataset,
                                        num_data=args.num_data,
                                        batch_size=args.batch_size,
                                        path='/rscratch/data/imagenet12/')
print('Data loaded')

if not args.quantize:
    model = ptcv_get_model(args.model, pretrained=True).cuda()
    model = nn.DataParallel(model)
    if args.train:
        train(model, train_loader, test_loader, args.epoch, args.work_dir,
              args.lr)
    else:
        acc = test(model, test_loader)
        print('model check pass with acc = {}', format(acc))
    print(model)
else:
    model = ptcv_get_model(args.model, pretrained=True)
    if 'shufflenet' in args.model:
        quantized_model = quantize_shufflenetv2(model,
                                                quant_conv=args.conv_bit,
                                                quant_act=args.act_bit,
                                                quant_mode=args.quant_mode,
                                                wt_per_channel=args.wc,
                                                wt_percentile=args.wp,
                                                act_percentile=args.ap).cuda()
    else:
        quantized_model = quantize_model(model,
                                         quant_conv=args.conv_bit,
                                         quant_act=args.act_bit,
                                         quant_mode=args.quant_mode,
                                         wt_per_channel=args.wc,
                                         wt_percentile=args.wp,
                                         act_percentile=args.ap).cuda()
    quantized_model = nn.DataParallel(quantized_model)
    if not (args.resume_from is None):
        quantized_model.load_state_dict(torch.load(args.resume_from),
                                        strict=False)

    print(quantized_model)
    if args.train:
        quantized_model.eval()
        train(quantized_model, train_loader, test_loader, args.epoch,
              args.work_dir, args.lr)
    else:
        if args.update:
            update(quantized_model, train_loader)

        quantized_model.eval()
        freeze_model(quantized_model)
        acc = test(quantized_model, test_loader)
        torch.save(quantized_model.state_dict(), 'test.pth')
        print('quantized model check pass with acc = {}'.format(acc))
    print(quantized_model)
