
from plugins.ithor_arm_plugin.ithor_arm_constants import ENV_ARGS
from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor, DepthSensorThor
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import ArmPointNavTaskSampler
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from projects.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import (
    ArmPointNaviThorBaseConfig,
)
from projects.armpointnav_baselines.experiments.armpointnav_mixin_ddppo import (
    ArmPointNavMixInPPOConfig,
)
from projects.armpointnav_baselines.experiments.armpointnav_mixin_simplegru import (
    ArmPointNavMixInSimpleGRUConfig,
)

class ArmPointNavRGBDepth(
    ArmPointNaviThorBaseConfig, ArmPointNavMixInPPOConfig, ArmPointNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        DepthSensorThor(
            height=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        RGBSensorThor(
            height=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
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
        self.ENV_ARGS = {**ENV_ARGS,  'renderDepthImage':True}

    @classmethod
    def tag(cls):
        return cls.__name__
