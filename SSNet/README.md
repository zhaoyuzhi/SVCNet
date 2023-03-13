# SVCNet (SSNet)

## 1 Train

Make sure you have installed the **svcnet** requirement and download the training data.

The joint training can be started by running:
```bash
sh runs/dv.sh
```

The training for both steps take approximately 8 days on a 8 NVIDIA Titan Xps or 6 days on a 8 NVIDIA V100s.

## 2 Validation

By default you can use **validation.py** to test the video colorization quality of trained models:
```bash
python validation.py
```
