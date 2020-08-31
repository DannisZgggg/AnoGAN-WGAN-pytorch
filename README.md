# AnoGAN-WGAN-pytorch
This is an UNOFFICIAL implement of  Pytorch based AnoGAN combined with WGAN


### Requirements

```
python>=3.6
torch==1.0.0
torchvision==0.2.2
albumentations>=0.4
```
### Note
- Current version has supported  training for mnist datasets.

### Visualization

There is inference results below. Left side is input and right side output.

After training digit "0", in inference stage, the model can recontruct well to "0" but fail to recontruct "8" which is abnormal input.

![00001_TP](https://github.com/DannisZgggg/AnoGAN-WGAN-pytorch/blob/master/resources/00001_TP.jpg)

![00002_TN](https://github.com/DannisZgggg/AnoGAN-WGAN-pytorch/blob/master/resources/00002_TN.jpg)