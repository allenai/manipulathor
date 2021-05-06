import gym
from torch import nn

from ithor_arm.ithor_arm_constants import ENV_ARGS
from ithor_arm.ithor_arm_sensors import (
    DistanceAgentArmToObjectSensor,
    DistanceObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor,
)
from ithor_arm.ithor_arm_task_samplers import ArmPointNavTaskSampler
from manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_mixin_ddppo import (
    ArmPointNavMixInPPOConfig,
)
from manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_mixin_simplegru import (
    ArmPointNavMixInSimpleGRUConfig,
)
from manipulathor_baselines.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import (
    ArmPointNaviThorBaseConfig,
)
from manipulathor_baselines.armpointnav_baselines.models.distance_only_model import DistanceOnlyModel


class DistanceOnlyArmPointNavDepth(
    ArmPointNaviThorBaseConfig,
    ArmPointNavMixInPPOConfig,
    ArmPointNavMixInSimpleGRUConfig,
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
        DistanceAgentArmToObjectSensor(),
        DistanceObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = ArmPointNavTaskSampler

    def __init__(self):
        super().__init__()

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )
        self.ENV_ARGS = {**ENV_ARGS, "renderDepthImage": True}


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return DistanceOnlyModel(
            action_space=gym.spaces.Discrete(len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

    @classmethod
    def tag(cls):
        return cls.__name__
