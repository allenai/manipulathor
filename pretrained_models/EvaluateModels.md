# Evaluating the Pre-trained Models
## Download the Weights
You can download the weights from <a href="https://drive.google.com/file/d/1wZi_IL5d7elXLkAb4jOixfY0M6-ZfkGM/view?usp=sharing">here</a>. After downloading the weights, extract them to `pretrained_models/saved_checkpoints`.
## Evaluating the Trained Models
### Table 1
#### Test-SeenObj
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_depth --mode test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```
#### Test-NovelObj
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ test_NovelObj_armpointnav_depth --mode test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```
#### SeenScenes-NovelObj
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ test_SeenScenes_NovelObj_armpointnav_depth --mode test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```

### Table 2

#### No-Vision
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_no_vision --mode test -c pretrained_models/saved_checkpoints/no_vision_armpointnav.pt
```

#### Disjoint Model
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_disjoint_depth --mode test -c pretrained_models/saved_checkpoints/disjoint_model_armpointnav.pt
```

#### RGB Model
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_rgb --mode test -c pretrained_models/saved_checkpoints/rgb_armpointnav.pt
```

#### RGB-Depth Model
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_rgbdepth --mode test -c pretrained_models/saved_checkpoints/rgbdepth_armpointnav.pt
```


#### Depth Model
```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ armpointnav_depth --mode test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```