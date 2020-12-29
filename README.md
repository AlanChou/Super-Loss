# Super Loss 
 
This is the unofficial PyTorch implementation of the paper "SuperLoss: A Generic Loss for Robust Curriculum Learning" in NIPS 2020.<br> 
http://papers.neurips.cc/paper/9289-data-parameters-a-new-family-of-parameters-for-learning-a-differentiable-curriculum.pdf


## Overview

This is a simple implementation of the paper where only image classification task on CIFAR is implemented. Note that in order to save time, the codebase is mostly based on my previous implementation of Truncated Loss.
As a result, some settings might be different from the paper and the file SuperLoss.py is "hard coded" which is not very ideal if one want to plug other loss functions (e.g. MSE or Focal Loss) into Super Loss.

I'm not sure if the code is 100% correct so if you find anything wrong, please don't hesitate to correct it. 
One may easily plug the SuperLoss.py file to other repo such as https://github.com/apple/ml-data-parameters to verify the correctness of the implementation. 
In general, use this repo as a simple playground and use the code at your own risk.


## Dependencies
This code is based on Python 3.5, with the main dependencies being PyTorch==1.2.0 torchvision==0.4.0 Additional dependencies for running experiments are: numpy, argparse, os, csv, sys, PIL, scipy


Run the code with the following example commands:<br>
###  Uniform Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --schedule 40 70 --epochs 100
```

## Comparison to Truncated Loss
Based on my few experiments on CIAFAR-10, it appears to me that it does not significantly outperform Truncated Loss (it may be due to the choice of hyper-parameters since I didn't spend much tuning it for SuperLoss.) 
Note that the paper of SuperLoss evalutated their own method with WideResNet28-10 while Truncated Loss was evalutated on ResNet-34 in the original paper. So IF the author of SuperLoss didn't re-evaluated Truncated Loss on WideResNet28-10 again by themselves and simply copied the reported performance in the Truncated Loss paper, then it's not a fair comparison which may explain my results that they perform similarly.)

## Best Accuracy on CIFAR-10 with Uniform Noise(noise rate = 0.4)
| Model             | Super Loss   | Truncated Loss |
| ----------------- | -----------  | -----------    | 
| [ResNet34]        |    88.4%  (this repo)  | 87.62% (reported by original paper)|

Despite the performance doesn't improve much in my case, it seems to maintain the accuracy better compares to Truncated Loss. 
The last epoch accuracies reported by Truncated Loss is 79% while it is 84% for SuperLoss.




