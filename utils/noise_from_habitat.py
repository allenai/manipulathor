#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Parameters contributed from PyRobot
https://pyrobot.org/
https://github.com/facebookresearch/pyrobot
Please cite PyRobot if you use this noise model
"""

from typing import Any, List, Optional, Sequence, Tuple

import attr
import numpy as np
import scipy.stats
from attr import Attribute
from numpy import ndarray

@attr.s(auto_attribs=True, init=False, slots=True)
class _TruncatedMultivariateGaussian:
    mean: np.ndarray
    cov: np.ndarray

    def __init__(self, mean: Sequence, cov: Sequence) -> None:
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        if len(self.cov.shape) == 1:
            self.cov = np.diag(self.cov)

        assert (
                np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
        ), "Only supports diagonal covariance"

    def sample(
            self,
            truncation: Optional[
                List[Optional[Tuple[Optional[Any], Optional[Any]]]]
            ] = None,
    ) -> ndarray:
        if truncation is not None:
            assert len(truncation) == len(self.mean)

        sample = np.zeros_like(self.mean)
        for i in range(len(self.mean)):
            stdev = np.sqrt(self.cov[i, i])
            mean = self.mean[i]
            # Always truncate to 3 standard deviations
            a, b = -3, 3

            if truncation is not None and truncation[i] is not None:
                trunc = truncation[i]
                if trunc[0] is not None:
                    a = max((trunc[0] - mean) / stdev, a)
                if trunc[1] is not None:
                    b = min((trunc[1] - mean) / stdev, b)

            sample[i] = scipy.stats.truncnorm.rvs(a, b, mean, stdev)

        return sample


@attr.s(auto_attribs=True, slots=True)
class MotionNoiseModel:
    linear: _TruncatedMultivariateGaussian
    rotation: _TruncatedMultivariateGaussian


@attr.s(auto_attribs=True, slots=True)
class ControllerNoiseModel:
    linear_motion: MotionNoiseModel
    rotational_motion: MotionNoiseModel
