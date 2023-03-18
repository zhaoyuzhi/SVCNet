# SVCNet (SSNet)

## 1 Train

Make sure you have installed the **svcnet** requirement and download the training data.

We omit the pre-training codes for `Refinement Module` and `Super-resolution Module` due to simplicity. Instead, we provide the [pre-trained models]() of them for users to reproduce the training of the full SSNet.

The joint training can be started by running:
```bash
sh runs/dv.sh
```

The training for both steps take approximately 8 days on 8 NVIDIA Titan Xp GPUs or 6 days on 8 NVIDIA V100 GPUs.

## 2 Validation

You can download the pre-trained CPNet model via this [link]() and the pre-trained SSNet model via this [link]().

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

You can download the pre-trained CPNet model via this [link]() and the pre-trained SSNet model via this [link]().

By default you can use **test.py** to test the video colorization quality of trained models:
```bash
python test.py
```

We have included a video sample and you can fully re-produce the results given in the folder.
