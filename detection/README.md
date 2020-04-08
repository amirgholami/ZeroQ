# ZeroQ: A Novel Zero Shot Quantization Framework (Object Detection)

This repository contains the PyTorch implementation for the object detetcion part for **CVPR 2020** paper [*ZeroQ: A Novel Zero-Shot Quantization Framework*](https://arxiv.org/abs/2001.00281).

This repository is adopted from [*mmdetection*](https://github.com/open-mmlab/mmdetection) repo.

## Installation

Please follow the [instruction of mmdetection](https://https://github.com/amirgholami/ZeroQ/blob/master/detection/docs/INSTALL.md) to install the dependencies and prepare the dataset.

We use RetinaNet with FPN and ResNet50 as backbone as our test model, which can be downloaded from [retinanet_r50_fpn](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth).

The folder structures should be the same as following

```
zeroQ_detection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── checkpoints
│   ├── retinanet_r50_fpn_2x_20190616-75574209.pth
```



Afterwards you can test ZeroQ for object detction with W8A8 by running

```bash
# generating distilled data
python tools/distill_data.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_2x_20190616-75574209.pth

# test the W8A8 quantized model
python tools/test.py configs/retinanet_r50_fpn_1x.py checkpoints/retinanet_r50_fpn_2x_20190616-75574209.pth --out=out.pkl --eval=bbox
```



Below is the results that you should get for 8-bit quantization (**W8A8** refers to the quantizing model to 8-bit weights and 8-bit activations).

| Models                                        | Single Precision mAP | W8A8 Top-1 |
| --------------------------------------------- | -------------------- | ---------- |
| [RetinaNet](https://arxiv.org/abs/1512.03385) | 36.4                 | 36.4       |



## Citation

ZeroQ has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the implementation useful for your work:

Y. Cai, Z. Yao, Z. Dong, A. Gholami, M. W. Mahoney, K. Keutzer. *ZeroQ: A Novel Zero Shot Quantization Framework*, CVPR 2020 [[PDF](https://arxiv.org/pdf/2001.00281.pdf)].