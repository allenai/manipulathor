# import platform
# import yaml

# from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
# from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_wide \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_narrow \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV
    # from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor


class ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy(
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV):

    def __init__(self):
        super().__init__()
        self.ENV_ARGS['motion_noise_type'] = 'habitat'
        self.ENV_ARGS['motion_noise_args'] = dict()
        self.ENV_ARGS['motion_noise_args']['multiplier_means'] = [1,1,1,1,1,1]
        self.ENV_ARGS['motion_noise_args']['multiplier_sigmas'] = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.ENV_ARGS['motion_noise_args']['effect_scale'] = .1 # .1 for eval .25 for training. change to .1 for fine-tune with action penalties

        self.ENV_ARGS['returnToStart'] = False # for eval