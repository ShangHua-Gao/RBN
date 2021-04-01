# Representative Batch Normalization (RBN) with Feature Calibration
The official implemention of the CVPR2021 oral paper: Representative Batch Normalization with Feature Calibration

**You only need to replace the BN with our RBN without any other adjustment.**

## Update
- 2021.4.1 The training code of ImageNet classification using RBN is released.

## Introduction
Batch Normalization (BatchNorm) has become the default component in modern neural networks to stabilize
training. In BatchNorm, centering and scaling operations,
along with mean and variance statistics, are utilized for
feature standardization over the batch dimension. The
batch dependency of BatchNorm enables stable training
and better representation of the network, while inevitably
ignores the representation differences among instances. We
propose to add a simple yet effective feature calibration
scheme into the centering and scaling operations of BatchNorm, enhancing the instance-specific representations with
the negligible computational cost. The centering calibration strengthens informative features and reduces noisy features. The scaling calibration restricts the feature intensity to form a more stable feature distribution. Our proposed variant of BatchNorm, namely Representative BatchNorm, can be plugged into existing methods to boost the
performance of various tasks such as classification, detection, and segmentation.


## Applications

### ImageNet classification
The training code of ImageNet classification is released in `ImageNet_training` folder.




## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{gao2021rbn,
  title={Representative Batch Normalization with Feature Calibration},
  author={Gao, Shang-Hua and Han, Qi and Li, Duo and Peng, Pai and Cheng, Ming-Ming and Pai Peng},
  booktitle=CVPR,
  year={2021}
}
```
## Contact
If you have any questions, feel free to E-mail Shang-Hua Gao (`shgao(at)live.com`) and Qi Han(`hqer(at)foxmail.com`).
