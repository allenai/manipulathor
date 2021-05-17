from ithor_arm.ithor_arm_constants import ENV_ARGS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_task_samplers import EasyArmPointNavTaskSampler
from manipulathor_baselines.arm_less_sensor_baselines.experiments.ithor.pred_distance_ithor_base import PredDistanceiThorBaseConfig
from manipulathor_baselines.arm_less_sensor_baselines.experiments.pred_distance_mixin_ddppo import PredDistanceMixInPPOConfig
from manipulathor_baselines.arm_less_sensor_baselines.experiments.pred_distance_mixin_simplegru import PredDistanceMixInSimpleGRUConfig


class PredDistanceDepth(
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

    #TODO maybe try only one room later

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
