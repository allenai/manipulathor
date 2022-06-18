"""Utility classes and functions for noise models used by the platforms."""
from typing import Tuple, Dict, List, Set, Union, Any, Optional

from utils.noise_from_habitat import MotionNoiseModel, _TruncatedMultivariateGaussian
import numpy as np

class NoiseInMotionHabitatFlavor:
    # Sample usage in __init__ for a config file:
        # self.ENV_ARGS['motion_noise_type'] = 'habitat'
        # self.ENV_ARGS['motion_noise_args'] = dict()
        # self.ENV_ARGS['motion_noise_args']['multiplier_means'] = [1,1,1,1,1,1]
        # self.ENV_ARGS['motion_noise_args']['multiplier_sigmas'] = [0,0,0,0,0,0,0]
        # self.ENV_ARGS['motion_noise_args']['effect_scale'] = 1
        
    def __init__(self,
            multiplier_means: List[float] = [1,1,1,1,1,1],
            multiplier_sigmas: List[float] = [0,0,0,0,0,0,0],
            effect_scale: float = 1,
        ) -> None:

        self.multiplier_means = multiplier_means
        self.multiplier_sigmas = multiplier_sigmas
        self.effect_scale = effect_scale
        
        self.ahead = None
        self.rotate = None

        self.reset_noise_model()
    
    def generate_linear_submodel(self,m1,m2,m3,s1,s2,s3):
        return MotionNoiseModel(
            _TruncatedMultivariateGaussian([m1 * 0.074, m2 * 0.036], [s1 * 0.019, s2 * 0.033]),
            _TruncatedMultivariateGaussian([m3 * 0.189], [s3 * 0.038]))
        
    def generate_rotational_submodel(self,m1,m2,m3,s1,s2,s3):
        return MotionNoiseModel(
            _TruncatedMultivariateGaussian([m1 * 0.002, m2 * 0.003], [s1 * 0.0, s2 * 0.002]),
            _TruncatedMultivariateGaussian([m3 * 0.219], [s3 * 0.019]))
    
    def generate_model_multipliers(self):
        return [np.random.normal(self.multiplier_means[i],self.multiplier_sigmas[i]) 
                for i in range(len(self.multiplier_means))]
    
    def reset_noise_model(self):
        model_multipliers = self.generate_model_multipliers()
        
        self.ahead = self.generate_linear_submodel(*model_multipliers)
        self.rotate = self.generate_rotational_submodel(*model_multipliers)
    
    def get_ahead_drift(self,*_):
        # returns [ahead change, left/right change, rot change] from an ahead command in [m,m,deg]
        
        if self.effect_scale == 0: #speedup for trivial case
            return [0,0,0]
        
        rotation_drift = self.effect_scale * self.ahead.rotation.sample()
        linear_drift = self.effect_scale * self.ahead.linear.sample()
        return [linear_drift[0], linear_drift[1], rotation_drift[0] * 180 / np.pi]

    def get_rotate_drift(self):
        # returns [ahead change, left/right change, rot change] from an ahead command in [m,m,deg]

        if self.effect_scale == 0: #speedup for trivial case
            return [0,0,0]
        
        rotation_drift = self.effect_scale * self.rotate.rotation.sample()
        linear_drift = self.effect_scale * self.rotate.linear.sample()

        return [linear_drift[0], linear_drift[1], rotation_drift[0] * 180 / np.pi]


class NoiseInMotionSimple1DNormal:
    # Sample usage in __init__ for a config file:
        # self.ENV_ARGS['motion_noise_type'] = 'simple1d'
        # self.ENV_ARGS['motion_noise_args'] = dict()
        # self.ENV_ARGS['motion_noise_args']['ahead_noise_meta_dist_params'] = {'bias_dist': [0,0.04], 'variance_dist': [0,0.10]}
        # self.ENV_ARGS['motion_noise_args']['lateral_noise_meta_dist_params'] = {'bias_dist': [0,0.04], 'variance_dist': [0,0.04]}
        # self.ENV_ARGS['motion_noise_args']['turning_noise_meta_dist_params'] = {'bias_dist': [0,10], 'variance_dist': [0,10]}

    def __init__(self,
            ahead_noise_meta_dist_params: Dict = {'bias_dist': [0,0], 'variance_dist': [0,0]},
            lateral_noise_meta_dist_params: Dict= {'bias_dist': [0,0], 'variance_dist': [0,0]},
            turning_noise_meta_dist_params: Dict = {'bias_dist': [0,0], 'variance_dist': [0,0]},
            effect_scale: float = 1,
        ) -> None:

        self.ahead_noise_meta_dist_params = ahead_noise_meta_dist_params
        self.lateral_noise_meta_dist_params = lateral_noise_meta_dist_params
        self.turning_noise_meta_dist_params = turning_noise_meta_dist_params
        self.effect_scale = effect_scale
        
        self.ahead_noise_params = [0,0]
        self.lateral_noise_params = [0,0]
        self.turning_noise_params = [0,0]

        self.reset_noise_model()
    
    def generate_motion_noise_params(self,meta_dist):
        bias = np.random.normal(*meta_dist['bias_dist'])
        variance = np.abs(np.random.normal(*meta_dist['variance_dist']))
        return [bias,variance]

    def reset_noise_model(self):
        self.ahead_noise_params = self.generate_motion_noise_params(self.ahead_noise_meta_dist_params)
        self.lateral_noise_params = self.generate_motion_noise_params(self.lateral_noise_meta_dist_params)
        self.turning_noise_params = self.generate_motion_noise_params(self.turning_noise_meta_dist_params)
    
    def get_ahead_drift(self, nominal_ahead):
        # returns [ahead change, left/right change, rot change] from an ahead command in [m,m,deg]
        linear_drift = self.effect_scale * np.random.normal(*self.ahead_noise_params)
        side_drift = self.effect_scale * np.random.normal(*self.lateral_noise_params)
        rotation_drift = np.arctan2(side_drift,linear_drift + nominal_ahead) * 180 / np.pi

        return [linear_drift, side_drift, rotation_drift ]

    def get_rotate_drift(self):
        # for this model, rotating incurs no positional drift. Returns [0,0,deg]
        rotation_drift = self.effect_scale * np.random.normal(*self.turning_noise_params)

        return [0, 0, rotation_drift]
