from legacy.rgbd_gt_mask_only_close import NoPickUpRGBDMaskOnlyClose


class VisualizeTrainMaskClose(
    NoPickUpRGBDMaskOnlyClose
):
    NUMBER_OF_TEST_PROCESS = 1
    VISUALIZE = True
    # TEST_SCENES = NoPickUpRGBDMaskOnlyClose.TRAIN_SCENES TODO
