import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler, WPickupAndExploreBOTS
from manipulathor_baselines.bring_object_baselines.experiments.ithor.temp_no_pu_no_noise_query_obj_w_gt_mask_and_rgb import NoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject

class RGBDWithPickupMoreExploreExp(
    NoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject
):
    TASK_SAMPLER = WPickupAndExploreBOTS



    def __init__(self):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "pickup_success_reward": 5.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,  # we are not using this
            "failed_action_penalty": -0.03,
            'exploration_reward': 0.05, #TODO is this a good value?
        }
