# SVCNet (CPNet)

## 1 Train

Make sure you have installed the **svcnet** environment and download the training data.

The warm-up training includes two steps. You can start it on the ImageNet dataset by running:
```bash
sh run_1st.sh
```

After the 1st step is done, running the following code for 2nd step on the DAVIS-Videvo dataset:
```bash
sh run_2nd.sh
```

The training for both steps take approximately 10 days on 8 NVIDIA Titan Xp GPUs.

## 2 Validation

### 2.1 Basic usage

You can download the pre-trained CPNet model via this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EXUIeeSbnqVJq7s4PU2emwABWfxLP1UKDHajSv9lGVH_3Q?e=q4aa8g). Please generate color scribbles before validation and you can find some useful scripts under **GCS** folder.

By default you can use **validation.py** to validate the single image colorization quality of trained models on DAVIS-Videvo dataset validation set:
```bash
python validation.py
```

Some parameters are concluded as:

- pre_train_cpnet_type: network architecture

- tag: (only valid for video validation) DAVIS or videvo

- save_rgb_path: a path to save the generated images

- finetune_path: the path to the pre-trained CPNet

- base_root: the path to input validation images

- scribble_root: the path to input color scribbles

- crop_size_h / crop_size_w: validation resolution

### 2.2 Other validation scripts

We include more validation scripts in **valers** sub-folder:

- Validate the pre-trained models given fix color scribbles or fixed number of random color scribbles:

| Name | Validation set | Scribble |
| ---- | :----: | :----: |
| validation_given_scribble_DAVIS_videvo | DAVIS+Videvo | √ |
| validation_given_scribble_ImageNet | ImageNet | × |
| validation_random_scribble_DAVIS_videvo | DAVIS+Videvo | √ |
| validation_random_scribble_ImageNet | ImageNet | × |

- Validate the pre-trained model from a number of 0-40 random color scribbles:

| Name | Validation set | Scribble |
| ---- | :----: | :----: |
| validation_sweep_DAVIS_videvo_without_resize | DAVIS+Videvo | × |
| validation_sweep_DAVIS_videvo | DAVIS+Videvo | × |
| validation_sweep_ImageNet | ImageNet | × |
