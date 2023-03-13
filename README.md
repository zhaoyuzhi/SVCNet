# SVCNet

supplementary material for paper "SVCNet: Real-time Scribble-based Video Colorization with Pyramid Networks", under review of IEEE Transactions on Image Processing

SVCNet is an architecture for scribble-based video colorization, which includes two sub-networks: CPNet and SSNet.

![network](./assets/pipeline.png)

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



### 1.3 Dataset

- image files: https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EYorh60RjVBEqnSJ_7tdBVMB6_Glq3b2vNk-UBXf9LpBTQ?e=MAf4lc

- lmdb files: https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/Ed0C2MlTsBdGov_bcszB-DsBpHRyZ1ZS_ApvhRkk1sbRQw?e=QaYg9M

## 2 Arrangement

- **CPNet** sub-folder includes scripts and codes for training and validating CPNet

- **SSNet** sub-folder includes scripts and codes for training and validating SSNet

- **Evaluation** sub-folder includes codes for evaluation (e.g., Tables II, IV, and V in the paper)

- **CS** sub-folder includes codes for generating validation scribbles

## 3 Visualization

A few video samples are illustrated below:

![network](./assets/gold-fish.gif)

![network](./assets/horsejump-high.gif)

![network](./assets/kite-surf.gif)
