# SVCNet (CPNet)

## 1 Train

Make sure you have installed the **svcnet** requirement and download the training data.

The warm-up training includes two steps. You can start the training by running:
```bash
sh run_1st.sh
```

After the 1st training is done, simply running:
```bash
sh run_2nd.sh
```

The training for both steps take approximately 10 days on a 8-GPU machine (NVIDIA Titan Xp).

## 2 Validation

By default you can use **validation.py** to test the single image colorization quality of trained models:
```bash
python validation.py
```
