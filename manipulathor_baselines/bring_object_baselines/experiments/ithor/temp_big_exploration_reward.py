from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler, NoPickupExploreBOTS

from manipulathor_baselines.bring_object_baselines.experiments.ithor.rgbd_gt_mask import RGBDGtMaskNoNoise


class BigBigExploreRGBDNOPickupMoreExploreExp(
    RGBDGtMaskNoNoise
):
    TASK_SAMPLER = NoPickupExploreBOTS

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.5

