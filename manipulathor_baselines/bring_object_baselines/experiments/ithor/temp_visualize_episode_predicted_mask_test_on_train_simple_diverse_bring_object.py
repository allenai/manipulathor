import platform

import gym
from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
import torch.optim as optim
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ithor_arm.bring_object_sensors import TargetObjectBBox, TargetLocationBBox, CategorySampleSensor, NoGripperRGBSensorThor, NoisyObjectMask, TempAllMasksSensor, TempObjectCategorySensor, TempEpisodeNumber
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.predicted_mask_simple_diverse_bring_object import PredictedMaskSimpleDiverseBringObject
from manipulathor_baselines.bring_object_baselines.experiments.ithor.predicted_mask_test_on_train_simple_diverse_bring_object import TestOnTrainPredictedMaskSimpleDiverseBringObject
from manipulathor_baselines.bring_object_baselines.experiments.ithor.temp_predict_mask_no_pu_no_noise_query_obj_w_gt_mask_and_rgb import PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject
from manipulathor_baselines.bring_object_baselines.losses.bring_object_losses import MaskLoss
from manipulathor_baselines.bring_object_baselines.models.pickup_object_with_mask_model import PickUpWMaskBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.predict_mask_small_bring_object_model import SmallBringObjectWPredictMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.small_bring_object_with_mask_model import SmallBringObjectWMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.small_depth_pickup_object_with_mask_model import SmallPickUpWMaskDepthBaselineActorCritic
from manipulathor_utils.debugger_util import ForkedPdb


class LongTestOnTrainPredictedMaskSimpleDiverseBringObject(
    PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""


    # NUMBER_OF_TEST_PROCESS = 1
    # VISUALIZE = True
    VISUALIZE = False
    TEST_SCENES = PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject.TRAIN_SCENES