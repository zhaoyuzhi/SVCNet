# SVCNet

supplementary material for paper "SVCNet: Real-time Scribble-based Video Colorization with Pyramid Networks", under review of IEEE Transactions on Image Processing.

SVCNet is an architecture for scribble-based video colorization, which includes two sub-networks: CPNet and SSNet.

![pipeline](./assets/pipeline.png)

## 1 Preparation

## 1.1 Environment

Basic requirements:
- pytorch==1.2.0 (higher version is also compatible)
- torchvision==0.4.0 (higher version is also compatible)
- cupy (we test the code with CUDA 10.0, and cupy version **cupy-cuda100** works well)
- python-opencv
- scipy
- scikit-image

If you use **conda**, the following command is helpful:
```bash
conda env create -f environment.yaml
conda activate svcnet
```

### 1.2 Pre-trained models

We upload the pre-trained SVCNet modules (including CPNet and SSNet) and other public pre-trained models (including PWCNet and VGG-16).

### 1.3 Dataset

We use [ImageNet](https://image-net.org/index.php), [DAVIS](https://davischallenge.org/), and [Videvo](https://github.com/phoenix104104/fast_blind_video_consistency) datasets as our training set. Please cite the original papers if you use these datasets.

#### 1.3.1 Training set of ImageNet (256x256)

- [JPG format](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/ERbTQ-SsaJJIrF975FHkX8IBsHRQhFucCaMxnW0cxUZzJg?e=M1j6eo)

- [saliency map files](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EcYyMwjOkrZOuG-JX6hmdrQBzvnn4s_PwLwqdyrVg701sQ?e=RIOu3s)

#### 1.3.2 Validation set of ImageNet (256x256)

- [JPG format](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EYOPzwZ0L-5HodA2uDZoUhsB90JhAWIyIYOwCwMSOHON1Q?e=tzbVI1)

- [saliency map files](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EZSybNec0IZDtCk1C1Gx3IsBN-Q1oyUkmKk1HfuDr2_f0g?e=MeBx9u)

#### 1.3.3 Training set of DAVIS-Videvo dataset

- [JPG format](ht)

- [lmdb format](http)

- [segmentation and saliency map files]()

#### 1.3.4 Validation set of DAVIS-Videvo dataset

- [JPG format](httpsf4lc)

- [segmentation and saliency map files]()

We generate saliency maps for images in the ImageNet dataset. Note that, images in the DAVIS dataset has segmentation labels, while we generate saliency maps for images in the Videvo dataset. The saliency detection method is ["Pyramid Feature Attention Network for Saliency detection"](https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection).

## 2 Arrangement

- **CPNet** sub-folder includes scripts and codes for training and validating CPNet

- **SSNet** sub-folder includes scripts and codes for training and validating SSNet

- **Evaluation** sub-folder includes codes for evaluation (e.g., Tables II, IV, and V in the paper)

- **CS** sub-folder includes codes for generating validation scribbles

## 3 Fast inference

We include a legacy video segment along with color scribble frames with 4 different styles. Users can easily reproduce the following results by:
```bash
cd SSNet
python test.py
```

## 4 Visualization

A few video samples are illustrated below:

![network](./assets/gold-fish.gif)

![network](./assets/horsejump-high.gif)

![network](./assets/kite-surf.gif)

## 5 Acknowledgement

Some codes are borrowed from the [SCGAN](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN), [VCGAN](https://github.com/zhaoyuzhi/VCGAN), and [DEVC](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) projects. Thanks for their awesome works.

## 6 Citation

If you think this work is helpful, please consider cite:
