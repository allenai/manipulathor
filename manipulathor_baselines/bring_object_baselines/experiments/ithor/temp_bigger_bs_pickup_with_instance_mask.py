import platform

import gym
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from ithor_arm.bring_object_sensors import DestinationObjectSensor, InitialObjectSensor, TargetObjectMask
from ithor_arm.bring_object_task_samplers import EasyPickUPObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pickup_object_with_mask_model import PickUpWMaskBaselineActorCritic


class BiggerBSPickUPObject(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

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
        #  add this back
        # DestinationObjectSensor(),
        # InitialObjectSensor(),
        PickedUpObjSensor(),
        TargetObjectMask(),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = EasyPickUPObjectTaskSampler
    NUM_PROCESSES = 40
    TRAIN_SCENES = ['FloorPlan1_physics']
    TEST_SCENES = ['FloorPlan1_physics']
    OBJECT_TYPES = [('Apple', 'Mug')]

    # if platform.system() == "Darwin":
    #     MAX_STEPS = 20 TODO remove?

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
        return PickUpWMaskBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

    @classmethod
    def tag(cls):
        return cls.__name__

    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 50
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
