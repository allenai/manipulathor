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
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
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
from manipulathor_baselines.bring_object_baselines.models.predict_simple_model_no_pu import PredictMaskSmallBringObjectWQueryObjRGBDModel


class PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""



    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=0, type='source', distance_thr=1.5)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=0, type='destination', distance_thr=1.5)
    source_object_category_sensor = TempObjectCategorySensor(type='source')
    destination_object_category_sensor = TempObjectCategorySensor(type='destination')
    category_sample_source = CategorySampleSensor(type='source')
    category_sample_destination = CategorySampleSensor(type='destination')
    rgb_for_detection_sensor = NoGripperRGBSensorThor(
        height=BringObjectiThorBaseConfig.SCREEN_SIZE,
        width=BringObjectiThorBaseConfig.SCREEN_SIZE,
        use_resnet_normalization=True,
        uuid="only_detection_rgb_lowres",
    )
    source_mask_sensor_prediction = PredictionObjectMask(type='source',object_query_sensor=category_sample_source, rgb_for_detection_sensor=rgb_for_detection_sensor)
    destination_mask_sensor_prediction = PredictionObjectMask(type='destination',object_query_sensor=category_sample_destination, rgb_for_detection_sensor=rgb_for_detection_sensor)


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
        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),
        category_sample_source,
        category_sample_destination,
        source_mask_sensor,
        destination_mask_sensor,
        source_mask_sensor_prediction,
        destination_mask_sensor_prediction,
    ]

    MAX_STEPS = 200

    # NUM_PROCESSES = 40
    NUM_PROCESSES = 20

    GRADIENT_STEPS = 64

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS

    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(type(self), self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:
            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PredictionObjectMask):
                    sensor_type.device = torch.device(kwargs["devices"][0])
        return sampler_args

    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(type(self), self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PredictionObjectMask):
                    sensor_type.device = torch.device(kwargs["devices"][0])

        return sampler_args

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
