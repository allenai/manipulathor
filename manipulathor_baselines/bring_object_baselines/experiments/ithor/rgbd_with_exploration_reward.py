import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask

from manipulathor_baselines.bring_object_baselines.experiments.ithor.rgbd_gt_mask import RGBDGtMaskNoNoise


class RGBDWithPickupMoreExploreExp(
    RGBDGtMaskNoNoise
):
    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = WPickUPExploreBringObjectTask


    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.05 #TODO is this a good value?
