import platform

import gym
import torch
from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, CategoryFeatureSampleSensor, RelativeArmDistanceToGoal, PreviousActionTaken, IsGoalObjectVisible
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PointNavEmulatorSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.losses.bring_object_losses import BinaryArmDistanceLoss
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_w_binary_head_model import RGBDModelWPointNavEmulatorWBinaryHead
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
import gym
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

class PointNavWBinaryHead(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?
    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)
    SENSORS = [
        RGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        PickedUpObjSensor(),
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),
        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),
        source_mask_sensor,
        destination_mask_sensor,
        PointNavEmulatorSensor(type='source', mask_sensor=source_mask_sensor),
        PointNavEmulatorSensor(type='destination', mask_sensor=destination_mask_sensor),
        # TempRealArmpointNav(uuid='point_nav_emul',type='source'),
        # TempRealArmpointNav(uuid='point_nav_emul', type='destination'),
        RelativeArmDistanceToGoal(),
        PreviousActionTaken(),
        IsGoalObjectVisible(),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    TASK_TYPE = ExploreWiseRewardTask

    NUM_PROCESSES = 20

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavWBinaryHead, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensor):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args
    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavWBinaryHead, self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensor):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return RGBDModelWPointNavEmulatorWBinaryHead(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE
        )
    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128 if platform.system() != "Darwin" else 5 #self.MAX_STEPS
        #
        save_interval = 500000  # from 50k
        log_interval = 1000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig), "binary_arm_dist": BinaryArmDistanceLoss()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                # PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
                PipelineStage(
                    loss_names=["ppo_loss", "binary_arm_dist"],
                    loss_weights=[1.0, 0.05], # TODO how is this?
                    max_stage_steps=ppo_steps,
                )
            ],

            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def tag(cls):
        return cls.__name__
