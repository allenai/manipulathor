import platform

import gym
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, TempAllMasksSensor, TempEpisodeNumber, TempObjectCategorySensor
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
from manipulathor_baselines.bring_object_baselines.models.rgbd_w_predict_mask_small_bring_object_model import PredictMaskSmallBringObjectWQueryObjRGBDModel


class PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    # NOISE_LEVEL = 0
    SENSORS = [
        NoGripperRGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="only_detection_rgb_lowres",
        ),
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
        UseCategoryFeatiureSensorAndChangeModel(),
        NoisyObjectMask(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=0.0, type='source', uuid='gt_mask_for_loss'),
        NoisyObjectMask(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=0.0, type='destination', uuid='gt_mask_for_loss'),
        TempAllMasksSensor(),
        TempEpisodeNumber(),
        TempObjectCategorySensor(type='source'),
        TempObjectCategorySensor(type='destination'),
    ]

    MAX_STEPS = 200

    # NUM_PROCESSES = 40
    NUM_PROCESSES = 20

    GRADIENT_STEPS = 64

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PredictMaskSmallBringObjectWQueryObjRGBDModel(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE,
        )

    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = self.GRADIENT_STEPS #self.MAX_STEPS
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
            named_losses={"ppo_loss": PPO(**PPOConfig)},#, "pred_distance_loss": PredictDistanceLoss()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                # PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
                PipelineStage(
                    loss_names=["ppo_loss"],#, "pred_distance_loss"],
                    loss_weights=[1.0],#, 1.0],
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

    #
