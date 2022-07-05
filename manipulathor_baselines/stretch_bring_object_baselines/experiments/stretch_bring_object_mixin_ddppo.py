import torch.optim as optim
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from torch.optim.lr_scheduler import LambdaLR

from manipulathor_baselines.stretch_bring_object_baselines.experiments.stretch_bring_object_base import \
    StretchBringObjectBaseConfig

from manipulathor_baselines.stretch_bring_object_baselines.models.pose_estimation_loss import PoseEstimationLoss
class StretchBringObjectMixInPPOConfig(StretchBringObjectBaseConfig):
    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128 #self.MAX_STEPS
        save_interval =  250000 # from 50k
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
            named_losses={"ppo_loss": PPO(**PPOConfig), 'pose_loss':PoseEstimationLoss()},#, "pred_distance_loss": PredictDistanceLoss()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                # PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
                PipelineStage(
                    loss_names=["ppo_loss", "pose_loss"],#, "pred_distance_loss"],
                    loss_weights=[1.0, 1.0],#, 1.0],
                    max_stage_steps=ppo_steps,
                )
            ],

            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
