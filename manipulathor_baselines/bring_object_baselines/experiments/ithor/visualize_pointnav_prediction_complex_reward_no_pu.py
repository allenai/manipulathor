import platform

import gym
import torch
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, TempAllMasksSensor, TempEpisodeNumber, TempObjectCategorySensor, CategoryFeatureSampleSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import MANIPULATHOR_ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PredictionObjectMask
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.pointnav_prediction_complex_reward_no_pu import PredictionPointNavNoPUNewModelAndHand
from manipulathor_baselines.bring_object_baselines.models.predict_simple_model_no_pu import PredictMaskSmallBringObjectWQueryObjRGBDModel


class VisualizePredictionPointNavNoPUNewModelAndHand(
    PredictionPointNavNoPUNewModelAndHand
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NUMBER_OF_TEST_PROCESS = 1
    VISUALIZE = True



