from abc import ABC

from ithor_arm.ithor_arm_constants import TRAIN_OBJECTS, TEST_OBJECTS
from legacy.arm_less_sensor_baselines.experiments import PredDistanceThorBaseConfig


class PredDistanceiThorBaseConfig(PredDistanceThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

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

    #
    # TRAIN_SCENES = [
    #     "FloorPlan{}_physics".format(str(i))
    #     for i in range(1, TOTAL_NUMBER_SCENES + 1)
    #     if (i % 3 == 1 or i % 3 == 0) and i != 28
    # ]  # last scenes are really bad
    # TEST_SCENES = [
    #     "FloorPlan{}_physics".format(str(i))
    #     for i in range(1, TOTAL_NUMBER_SCENES + 1)
    #     if i % 3 == 2 and i % 6 == 2
    # ]
    # VALID_SCENES = [
    #     "FloorPlan{}_physics".format(str(i))
    #     for i in range(1, TOTAL_NUMBER_SCENES + 1)
    #     if i % 3 == 2 and i % 6 == 5
    # ]

    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert (
        len(ALL_SCENES) == TOTAL_NUMBER_SCENES
        and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES
    )

    OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

    UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))
