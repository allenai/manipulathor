from typing import Dict, Tuple
import torch
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr

def calc_pose_estimation_loss(positions, rotations, odom_positions, odom_rotations):
    # Fixes zero crossings
    rotations[torch.where(rotations > 180)[0]] -= 360
    rotations[torch.where(rotations < -180)[0]] += 360

    unnormalized_pos_error = torch.mean(torch.abs(positions))
    unnormalized_rot_error = torch.mean(torch.abs(rotations))

    # Normalize based on max movement
    positions = positions / 0.2
    rotations = rotations / 30.0

    pos_error = torch.mean(torch.abs(positions))
    rot_error = torch.mean(torch.abs(rotations))


    odom_rotations[torch.where(odom_rotations > 180)[0]] -= 360
    odom_rotations[torch.where(odom_rotations < -180)[0]] += 360
    unnormalized_odom_pos_error = torch.mean(torch.abs(odom_positions))
    unnormalized_odom_rot_error = torch.mean(torch.abs(odom_rotations))

    pose_error = pos_error + rot_error
    return pose_error, {'pose_error': pose_error.item(),
                        'pos_error': pos_error.item(), 
                        'rot_error': rot_error.item(), 
                        'unnormalized_pos_error': unnormalized_pos_error.item(), 
                        'unnormalized_rot_error': unnormalized_rot_error.item(),
                        'unnormalized_odom_pos_error': unnormalized_odom_pos_error.item(),
                        'unnormalized_odom_rot_error': unnormalized_odom_rot_error.item()}

class PoseEstimationLoss(AbstractActorCriticLoss):
    """
    Loss for pose estimation
    """

    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        if 'pose_errors' not in actor_critic_output.extras or len(actor_critic_output.extras['pose_errors']) == 0:
            return torch.tensor(0.0), {}

        positions = torch.stack([actor_critic_output.extras['pose_errors'][i]['position'] for i in range(len(actor_critic_output.extras['pose_errors']))])
        rotations = torch.stack([actor_critic_output.extras['pose_errors'][i]['rotation'] for i in range(len(actor_critic_output.extras['pose_errors']))])


        odom_positions = torch.stack([actor_critic_output.extras['pose_errors'][i]['odom_pos'] for i in range(len(actor_critic_output.extras['pose_errors']))])
        odom_rotations = torch.stack([actor_critic_output.extras['pose_errors'][i]['odom_rot'] for i in range(len(actor_critic_output.extras['pose_errors']))])

        return calc_pose_estimation_loss(positions, rotations)
