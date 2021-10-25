from abc import ABC

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

    # NOTE @samir
    # TOTAL_NUMBER_SCENES = 120
    TOTAL_NUMBER_SCENES = 120

    TRAIN_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        # for i in range(1, 20 + 1)
        for i in tuple(list(range(1, 21)) + list(range(201, 221)) + list(range(301, 321)) + list(range(401, 421)))
    ]
    VALID_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        # for i in range(21, 26)
        for i in tuple(list(range(21, 26)) + list(range(221, 226)) + list(range(321, 326)) + list(range(421, 426)))
    ]
    TEST_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        # for i in range(26, 31)
        for i in tuple(list(range(26, 31)) + list(range(226, 231)) + list(range(326, 331)) + list(range(426, 431)))
    ]


    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert (
        len(ALL_SCENES) == TOTAL_NUMBER_SCENES
        and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES
    )

    OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

    UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))
