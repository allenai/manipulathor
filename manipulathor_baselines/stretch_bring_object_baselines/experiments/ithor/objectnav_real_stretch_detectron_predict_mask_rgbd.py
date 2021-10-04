import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import DepthSensorThor

from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.rgbd_w_predict_mask_small_bring_object_model import PredictMaskSmallBringObjectWQueryObjRGBDModel
from utils.stretch_utils.real_stretch_bring_object_task_sampler import RealStretchDiverseBringObjectTaskSampler
from utils.stretch_utils.real_stretch_sensors import StretchPickedUpObjSensor, StretchDetectronObjectMask, DepthSensorIntelRealStretch, RGBSensorIntelRealStretch
from utils.stretch_utils.real_stretch_tasks import StretchRealBringObjectTask, StretchRealObjectNavTask


class StretchPredictRGBDExp(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    SENSORS = [
        # NoGripperRGBSensorThor(
        #     height=BringObjectiThorBaseConfig.SCREEN_SIZE,
        #     width=BringObjectiThorBaseConfig.SCREEN_SIZE,
        #     use_resnet_normalization=True,
        #     uuid="only_detection_rgb_lowres",
        # ),
        RGBSensorIntelRealStretch(
            height=224, #TODO sure?
            width=224,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorIntelRealStretch(
            height=224,
            width=224,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        StretchPickedUpObjSensor(),
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),
        StretchDetectronObjectMask(noise=0, source_camera='intel', type='source'),
        StretchDetectronObjectMask(noise=0, source_camera='intel', type='destination'),
    ]

    MAX_STEPS = 200

    # POTENTIAL_VISUALIZERS = BringObjectiThorBaseConfig.POTENTIAL_VISUALIZERS + [MaskImageVisualizer]

    if platform.system() == "Darwin":
        MAX_STEPS = 200#3


    NUM_PROCESSES = 40

    # TRAIN_SCENES = ['FloorPlan1_physics']
    # TEST_SCENES = ['FloorPlan1_physics']
    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS

    TASK_SAMPLER = RealStretchDiverseBringObjectTaskSampler
    TASK_TYPE = StretchRealObjectNavTask



    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return SmallBringObjectWQueryObjGtMaskRGBDModel(
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
