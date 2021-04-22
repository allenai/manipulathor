# Evaluating the Pre-trained Models
## Download the Weights
You can download the weights from [here](https://drive.google.com/file/d/1axZRgY3oKgATu0zoi1LeLxUtqpIXmJCg/view?usp=sharing).
After downloading the weights, extract them to `pretrained_models/saved_checkpoints`.
## Evaluating the Trained Models
### Table 1
#### Test-SeenObj
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_depth.py -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```
#### Test-NovelObj
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/test_NovelObj_armpointnav_depth -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```
#### SeenScenes-NovelObj
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/test_SeenScenes_NovelObj_armpointnav_depth -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/depth_armpointnav.pt
```

### Table 2

#### No-Vision
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_no_vision -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/no_vision_armpointnav.pt
```

#### Disjoint Model
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_disjoint_depth -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/disjoint_model_armpointnav.pt
```

#### RGB Model
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_rgb -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/rgb_armpointnav.pt
```

#### RGB-Depth Model
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_rgbdepth -o test_out -s 1 -t test -c pretrained_models/saved_checkpoints/rgbdepth_armpointnav.pt
```


#### Depth Model
```
allenact manipulathor_baselines/armpointnav_baselines/experiments/ithor/armpointnav_depth -o test_out -s 1 -t test -cpretrained_models/saved_checkpoints/depth_armpointnav.pt
```