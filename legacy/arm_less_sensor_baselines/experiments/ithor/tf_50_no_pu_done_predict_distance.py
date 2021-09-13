import gym
from torch import nn

from ithor_arm.ithor_arm_constants import ENV_ARGS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_task_samplers import EasyArmPointNavTaskSampler
from legacy.arm_less_sensor_baselines.experiments import PredDistanceiThorBaseConfig
from legacy.arm_less_sensor_baselines.experiments import PredDistanceMixInPPOConfig
from legacy.arm_less_sensor_baselines.experiments import PredDistanceMixInSimpleGRUConfig
from legacy.arm_less_sensor_baselines.models.pred_distance_models import PredDistanceBaselineActorCritic


class HalfTFNoPUDonePredDistanceDepth(
    PredDistanceiThorBaseConfig,
    PredDistanceMixInPPOConfig,
    PredDistanceMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        DepthSensorThor(
            height=PredDistanceiThorBaseConfig.SCREEN_SIZE,
            width=PredDistanceiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        InitialAgentArmToObjectSensor(),
        InitialObjectToGoalSensor(),
        RelativeAgentArmToObjectSensor(),
        RelativeObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = EasyArmPointNavTaskSampler
    NUM_PROCESSES = 40


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
    def tag(cls):
        return cls.__name__

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PredDistanceBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            teacher_forcing=.5,
        )
