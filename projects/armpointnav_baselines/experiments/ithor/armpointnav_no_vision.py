import ai2thor
import gym

from plugins.ithor_arm_plugin.ithor_arm_constants import ENV_ARGS
from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor, DepthSensorThor, NoVisionSensorThor
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import ArmPointNavTaskSampler

from projects.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import (
    ArmPointNaviThorBaseConfig,
)
from projects.armpointnav_baselines.experiments.armpointnav_mixin_ddppo import (
    ArmPointNavMixInPPOConfig,
)
from projects.armpointnav_baselines.experiments.armpointnav_mixin_simplegru import (
    ArmPointNavMixInSimpleGRUConfig,
)
import torch.nn as nn

from projects.armpointnav_baselines.models.arm_pointnav_models import ArmPointNavBaselineActorCritic


class ArmPointNavNoVision(
    ArmPointNaviThorBaseConfig, ArmPointNavMixInPPOConfig, ArmPointNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        NoVisionSensorThor(
            height=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid="rgb_lowres",
        ),
        RelativeAgentArmToObjectSensor(),
        RelativeObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = ArmPointNavTaskSampler #

    def __init__(self):
        super().__init__()

        assert self.CAMERA_WIDTH == 224 and self.CAMERA_HEIGHT == 224 and self.VISIBILITY_DISTANCE == 1 and self.STEP_SIZE == 0.25
        self.ENV_ARGS = {**ENV_ARGS,  'renderDepthImage':False}


    @classmethod
    def tag(cls):
        return cls.__name__
