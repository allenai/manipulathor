from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask

from legacy.rgbd_gt_mask import RGBDGtMaskNoNoise


class RGBDWithPickupMoreExploreExp(
    RGBDGtMaskNoNoise
):
    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = WPickUPExploreBringObjectTask


    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.05 # is this a good value?
