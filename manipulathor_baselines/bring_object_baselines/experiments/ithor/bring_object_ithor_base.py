from abc import ABC

import torch

from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import BringObjectTask
from ithor_arm.ithor_arm_constants import TRAIN_OBJECTS, TEST_OBJECTS
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_thor_base import BringObjectThorBaseConfig


class BringObjectiThorBaseConfig(BringObjectThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = BringObjectTask

    NUM_PROCESSES = 40
    # add all the arguments here
    TOTAL_NUMBER_SCENES = 30

    TRAIN_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, 20 + 1)
    ]
    VALID_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(21, 26)
    ]
    TEST_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(26, 31)
    ]


    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert (
        len(ALL_SCENES) == TOTAL_NUMBER_SCENES
        and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES
    )

    OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

    UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))

    # TEST_GPU_IDS = list(range(torch.cuda.device_count()))
    # TEST_SCENES = BringObjectiThorBaseConfig.TEST_SCENES

    # what is the plan?
    # TEST_GPU_IDS = list(range(min(len(TEST_SCENES), torch.cuda.device_count())))
    # NUMBER_OF_TEST_PROCESS = len(TEST_SCENES)
    # print('TEST_GPU_IDS', TEST_GPU_IDS)
    TEST_GPU_IDS = list(range(min(len(TEST_SCENES), torch.cuda.device_count())))