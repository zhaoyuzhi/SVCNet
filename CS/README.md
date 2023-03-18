# Color Scribble Generation

## 1 Run

Make sure **svcnet** environment is installed.

Copy the **pwcNet-default.pytorch** file under this sub-folder, or change the **--pwcnet_path** parameter.

- Generate color scribbles for ImageNet images:
```bash
python generate_color_scribbles_ImageNet.py
```

- Generate color scribbles for DAVIS-Videvo frames:
```bash
python generate_color_scribbles_DAVIS_videvo.py
```

- Generate color scribbles for frames in arbitary videos:
```bash
python generate_color_scribbles_for_video.py
```

- Generate different color scribbles from a given scribble folder:
```bash
python mapping_diverse_color_scribbles.py
```

## 2 Visualization

Please run the following command to widen the generated color scribbles for better visualization:
```bash
python widen_color_scribbles_for_better_vis.py
```
