# Super Loss 
 
This is the unofficial PyTorch implementation of the paper "SuperLoss: A Generic Loss for Robust Curriculum Learning" in NIPS 2020.<br> 
https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf


## Overview

THE CURREENT CODE IS UNDER MAINTAINANCE AND IT IS NOT CORRECT. The labertw function should be implemented with PyTorch instead of using the scipy library as mentioned in https://github.com/AlanChou/Truncated-Loss/issues/3#issuecomment-753650227. I'll try to fix this when I'm available.

This is a simple implementation of the paper where only image classification task on CIFAR is implemented. Note that in order to save time, the codebase is mostly based on my previous implementation of Truncated Loss.
As a result, some settings might be different from the paper and the file SuperLoss.py is "hard coded" which is not very ideal if one want to plug other loss functions (e.g. MSE or Focal Loss) into Super Loss.


## Dependencies
This code is based on Python 3.5, with the main dependencies being PyTorch==1.2.0 torchvision==0.4.0 Additional dependencies for running experiments are: numpy, argparse, os, csv, sys, PIL, scipy


Run the code with the following example commands:<br>
###  Uniform Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --schedule 40 70 --epochs 100
```







