# ZeroQ: A Novel Zero Shot Quantization Framework



## Introduction

This repository contains the PyTorch implementation for the paper *ZeroQ: A Novel Zero-Shot Quantization Framework*. The released version currently supports both distilled data generation and uniform quantization to 8-bit weights/activations.

## Dependencies

- The code is based on PyTorch 1.2 (cuda10), you can follow the official instruction from [PyTorch](https://pytorch.org/) to install PyTorch

- Then install other dependencies through:

```bash
pip install -r requirements.txt --user
```

## Dataset

If you already have the ImageNet validation dataset for PyTorch, you can create a link to the data folder and use it:

```bash
# prepare dataset, change the path to your own. Here /path/to/imagenet/ is the original path to your ImageNet dataset, and data/ is the symbolic link.
ln -s /path/to/imagenet/ data/
```

The folder structures should be the same as followinng
```
zeroq
├── utils
├── data
│   ├── imagenet
│   │   ├── val
```

**Note that ZeroQ doesn't rely on data to fine-tune after quantization, so validation set from ImageNet would be enough for reproducing our results**

## Evaluate

- You can run the following bash file to reproduce all of our results for W8A8


```bash
bash run.sh
```

- You can also test a single model by

```bash
export CUDA_VISIBLE_DEVICES=0
python uniform_test.py [--dataset] [--model] [--batch_size] [--test_batch_size]

optional arguments:
--dataset                   type of dataset (default: imagenet)
--model                     model to be quantized (default: resnet18)
--batch-size                batch size of distilled data (default: 64)
--test-batch-size           batch size of test data (default: 512)
```

**W8A8** refers to the quantizing model to 8-bit weights and 8-bit activations.

*(0.1\% Top-1 Accuracy difference might come form variance of distilled data)*



| Models                                          | Orginal Top-1 Acc | W8A8 Top-1 Acc |
| ----------------------------------------------- | :---------------: | :------------: |
| [ResNet18](https://arxiv.org/abs/1512.03385)    |       71.47       |     71.43      |
| [ResNet50](https://arxiv.org/abs/1512.03385)    |       77.72       |     77.67      |
| [InceptionV3](https://arxiv.org/abs/1512.00567) |       78.88       |     78.72      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381) |       73.03       |     72.91      |
| [ShuffleNet](https://arxiv.org/abs/1707.01083)  |       65.07       |     64.94      |
| [SqueezeNext](https://arxiv.org/abs/1803.10615) |       69.38       |     69.17      |


## Citation
ZeroQ has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the implementation useful for your work:
Y. Cai, Z. Yao, Z. Dong, Amir Gholami, K. Keutzer, and M. W. Mahoney. ZeroQ: A Novel Zero Shot Quantization Framework, under review [PDF]()

