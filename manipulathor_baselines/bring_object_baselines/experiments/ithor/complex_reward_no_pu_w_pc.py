import platform

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_constants import FOV
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor, BinnedPointCloudMapTHORSensor, ReachableBoundsTHORSensor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor
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
from ithor_arm.pointcloud_sensors import KianaBinnedPointCloudMapTHORSensor, KianaReachableBoundsTHORSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.gt_mask_with_memory_model import MemoryWGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.mask_with_point_cloud_model import MaskWPointCloudSensor
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_utils.debugger_util import ForkedPdb


class ComplexRewardNoPUWPointCloudMemory(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?
    map_range_sensor = KianaReachableBoundsTHORSensor(margin=1.0)
    map_info = dict(
        map_range_sensor=map_range_sensor,
        # vision_range_in_cm=40 * 5,
        map_size_in_cm=1050,
        resolution_in_cm=20,
    )
    single_noisy_object_mask_source = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    single_noisy_object_mask_destination = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)

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
        UseCategoryFeatiureSensorAndChangeModel(),
        # NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr),
        # NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr),
        single_noisy_object_mask_source,
        single_noisy_object_mask_destination,
        KianaBinnedPointCloudMapTHORSensor(fov=FOV, ego_only=False, mask_sensor=single_noisy_object_mask_source,**map_info, type='source'),
        KianaBinnedPointCloudMapTHORSensor(fov=FOV, ego_only=False, mask_sensor=single_noisy_object_mask_destination,**map_info, type='destination'),
    ]

    MAX_STEPS = 200
    NUM_PROCESSES = 15

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = ExploreWiseRewardTask


    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS

    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(ComplexRewardNoPUWPointCloudMemory, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for binned_map_sensor in sampler_args['sensors']:
                if isinstance(binned_map_sensor, KianaBinnedPointCloudMapTHORSensor):
                    binned_map_sensor.device = torch.device(kwargs["devices"][0])

            # # print('KIANA: PROCEESSSSS', kwargs["process_ind"], len(kwargs["devices"]), kwargs["devices"])
            # # binned_map_sensor.device = torch.device(kwargs["process_ind"] % len(kwargs["devices"]))


            # binned_map_sensor = next(s for s in sampler_args["sensors"] if isinstance(s, KianaBinnedPointCloudMapTHORSensor))
            # binned_map_sensor.device = torch.device(kwargs["devices"][0])
        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?

        self.ENV_ARGS['visibilityDistance'] = self.distance_thr



    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MaskWPointCloudSensor(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE
        )

    @classmethod
    def tag(cls):
        return cls.__name__
