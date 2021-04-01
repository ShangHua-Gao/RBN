# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |



### ResNet50
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py\
    --model resnet50 --epochs 100
```
### ResNet101
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py\
    --model resnet101 --epochs 100
```


## Mixed precision training
Automatic Mixed Precision (AMP) training on GPU for Pytorch can be enabled with the [NVIDIA Apex extension](https://github.com/NVIDIA/apex).

Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput, generally without loss in model accuracy. Mixed precision training also often allows larger batch sizes. GPU automatic mixed precision training for Pytorch Vision can be enabled via the flag value `--apex=True`.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --model resnet50 --epochs 100 --apex
```
