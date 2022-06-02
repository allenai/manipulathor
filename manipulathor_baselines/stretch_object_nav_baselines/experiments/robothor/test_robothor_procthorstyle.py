import datasets
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_narrow \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_wide import \
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_for_procthor_clip_resnet50_rgb_only import \
    ProcTHORObjectNavClipResnet50RGBOnly
from utils.procthor_utils.procthor_object_nav_task_samplers import RoboThorObjectNavTestTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask, ObjectNavTask, StretchNeckedObjectNavTask

from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment


from manipulathor_utils.debugger_util import ForkedPdb



class ObjectNavTestProcTHORstyle(ProcTHORObjectNavClipResnet50RGBOnly):
    EVAL_TASKS = datasets.load_dataset(
        f"allenai/robothor-objectnav-eval", use_auth_token=True
    )

    TASK_SAMPLER = RoboThorObjectNavTestTaskSampler
    TASK_TYPE = StretchNeckedObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    TEST_ON_VALIDATION = True

    @classmethod
    def tag(cls):
        return super().tag() + "-RoboTHOR-Target"

    # def make_sampler_fn(self, task_sampler_args: TaskSamplerArgs, **kwargs):
    #     if task_sampler_args.mode == "eval":
    #         return ObjectNavTestTaskSampler(args=task_sampler_args)
    #     else:
    #         return ObjectNavTaskSampler(args=task_sampler_args)
    
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
        # ForkedPdb().set_trace()

        assert kwargs["sampler_mode"] == "eval"
        return cls.TASK_SAMPLER(**kwargs)

    def valid_task_sampler_args(self, **kwargs):
        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["validation"],
            mode="eval",
            max_tasks=150,
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
            max_tasks=150,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            **kwargs,
        )
        return out