# Color Scribble Generation

## 1 Run

Make sure you have installed the **svcnet** environment.

### 1.1 Run for image datasets

You can generate color scribbles for ImageNet images by:
```bash
python generate_color_scribbles_ImageNet.py
```
Actually, it adapts to arbitary image datasets.

### 1.2 Run for video datasets

Copy the **pwcNet-default.pytorch** file under this sub-folder, or change the **--pwcnet_path** parameter.

- Generate color scribbles for DAVIS-Videvo frames:
```bash
python generate_color_scribbles_DAVIS_videvo.py
```

- Generate color scribbles for arbitary video frames (a pre-defined color scribble initial image is needed):
```bash
python generate_color_scribbles_video.py
```

### 1.3 Generate diverse color scribbles

You can generate more different color scribbles from generated color scribbles:
```bash
python mapping_diverse_color_scribbles.py
```
We have already defined some transformation functions (mapping_func1 - mapping_func4) in the script and you can replace them.

## 2 Visualization

Please run the following command to widen the generated color scribbles for better visualization:
```bash
python widen_color_scribbles_for_better_vis.py
```
