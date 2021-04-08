
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import ArmPointNavTaskSampler

from projects.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import (
    ArmPointNaviThorBaseConfig,
)

from projects.armpointnav_baselines.experiments.ithor.armpointnav_depth import ArmPointNavDepth


class TestOnUObjUSceneRealDepthRandomAgentLocArmNav(
    ArmPointNavDepth
):

    MAX_STEPS = 200
    TASK_SAMPLER = ArmPointNavTaskSampler
    TEST_SCENES = ArmPointNaviThorBaseConfig.TEST_SCENES
    OBJECT_TYPES = ArmPointNaviThorBaseConfig.UNSEEN_OBJECT_TYPES
