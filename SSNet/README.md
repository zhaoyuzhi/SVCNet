# SVCNet (SSNet)

## 1 Train

Make sure you have installed the **svcnet** environment and download the training data.

We omit warm-up pre-training codes for **Refinement Module** and **Super-resolution Module** due to simplicity. Instead, we provide the [pre-trained models](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EoAhNnlzoe1LkoI0CSlw9q8B-zOmJlhZUoxBVC3m3eiDUQ?e=WeTQHQ) of them for users to reproduce the training of the full SSNet.

The joint training can be started by running:
```bash
sh runs/dv.sh
```

The training for both steps take approximately 8 days on 8 NVIDIA Titan Xp GPUs or 6 days on 8 NVIDIA V100 GPUs.

## 2 Validation

You can download the pre-trained CPNet model via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EXUIeeSbnqVJq7s4PU2emwABWfxLP1UKDHajSv9lGVH_3Q?e=q4aa8g) and the pre-trained SSNet model via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EaE2q8nnMv5Hv6qDJduc6_EB6VNe5DGwavHICUwwwlqS_A?e=V4zKII). Please generate color scribbles before validation and you can find some useful scripts under **GCS** folder.

By default you can use **validation.py** to validate the video colorization quality of trained models:
```bash
python validation.py
```

Some parameters are concluded as:

- pre_train_cpnet_type: network architecture

- pre_train_ssnet_type: network architecture

- tag: (only valid for video validation) DAVIS or videvo

- save_rgb_path: a path to save the generated images

- cpnet_path: the path to the pre-trained CPNet

- ssnet_path: the path to the pre-trained SSNet

- pwcnet_path: the path to the pre-trained PWCNet

- base_root: the path to input validation images

- scribble_root: the path to input color scribbles

- crop_size_h / crop_size_w: validation resolution

## 3 Test

You can download the pre-trained CPNet model via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EXUIeeSbnqVJq7s4PU2emwABWfxLP1UKDHajSv9lGVH_3Q?e=q4aa8g) and the pre-trained SSNet model via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EaE2q8nnMv5Hv6qDJduc6_EB6VNe5DGwavHICUwwwlqS_A?e=V4zKII). The grayscale testing video frames and color scribbles are included in **test_data** folder.

By default you can use **test.py** to test the video colorization quality of trained models:
```bash
python test.py
```
