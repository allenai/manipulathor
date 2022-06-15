import datasets
import torch
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_narrow \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_wide import \
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_for_procthor_clip_resnet50_rgb_only import \
    ProcTHORObjectNavClipResnet50RGBOnly
from utils.procthor_utils.procthor_object_nav_task_samplers import RoboThorObjectNavTestTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask, ObjectNavTask, StretchNeckedObjectNavTask, StretchNeckedObjectNavTaskUpdateOrder

from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment


from manipulathor_utils.debugger_util import ForkedPdb


# class ObjectNavRoboTHORTestProcTHORstyle(ProcTHORObjectNavClipResnet50RGBOnly):
class ObjectNavRoboTHORTestProcTHORstyle(ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV):
    EVAL_TASKS = datasets.load_dataset(
        f"allenai/robothor-objectnav-eval", use_auth_token=True
    )

    TEST_TASK_SAMPLER = RoboThorObjectNavTestTaskSampler
    TEST_ON_VALIDATION = True
    # TEST_GPU_IDS = list(range(torch.cuda.device_count())) # uncomment for vision server testing

    @classmethod
    def tag(cls):
        return super().tag() + "-RoboTHOR-Test"
    
    @classmethod
    def make_sampler_fn(cls, **kwargs):
        from datetime import datetime

        now = datetime.now()

        exp_name_w_time = cls.__name__ + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if cls.VISUALIZE:
            visualizers = [
                viz(exp_name=exp_name_w_time) for viz in cls.POTENTIAL_VISUALIZERS
            ]
            kwargs["visualizers"] = visualizers
        kwargs["exp_name"] = exp_name_w_time

        if kwargs["sampler_mode"] == "train":
            return cls.TASK_SAMPLER(**kwargs)
        else:
            return cls.TEST_TASK_SAMPLER(**kwargs)

    def valid_task_sampler_args(self, **kwargs):
        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["validation"],
            mode="eval",
            max_tasks=15,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            **kwargs,
        )
        return out

    def test_task_sampler_args(self, **kwargs):
        if self.TEST_ON_VALIDATION:
            return self.valid_task_sampler_args(**kwargs)

        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["test"].shuffle(),
            mode="eval",
            max_tasks=15,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            **kwargs,
        )
        return out