"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
from typing import Tuple, Optional

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from isdf.modules import trainer
from isdf.datasets.data_util import FrameData

from manipulathor_utils.debugger_util import ForkedPdb
from utils.model_utils import LinearActorHeadNoCategory
from utils.hacky_viz_utils import hacky_visualization
from utils.stretch_utils.stretch_thor_sensors import check_for_nan_visual_observations



class StretchObjectDisplacementISDFModel(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for preddistancenav task.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
        teacher_forcing=1,
        use_odom_pose=False,
        visualize=False,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.visualize = visualize

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size
        self.use_odom_pose = use_odom_pose

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 5,
                        'layer_channels': [32, 64, 32],
                        'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                        'strides': [(4, 4), (2, 2), (1, 1)],
                        'paddings': [(0, 0), (0, 0), (0, 0)],
                        'dilations': [(1, 1), (1, 1), (1, 1)],
                        'output_height': 24,
                        'output_width': 24,
                        'output_channels': 512,
                        'flatten': True,
                        'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)
        network_args = {'input_channels': 5,
                        'layer_channels': [32, 64, 32],
                        'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                        'strides': [(4, 4), (2, 2), (1, 1)],
                        'paddings': [(0, 0), (0, 0), (0, 0)],
                        'dilations': [(1, 1), (1, 1), (1, 1)],
                        'output_height': 24,
                        'output_width': 24,
                        'output_channels': 512,
                        'flatten': True,
                        'output_relu': True}
        self.full_visual_encoder_arm = make_cnn(**network_args)

        self.map_embedding = make_cnn(**network_args)

        # self.detection_model = ConditionalDetectionModel()
        self.body_pointnav_embedding = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
        )
        self.arm_pointnav_embedding = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
        )
        num_rnn_inputs = 4
        if use_odom_pose:
            self.odom_pose_embedding = nn.Sequential(
                nn.Linear(3, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 512),
            )
            num_rnn_inputs = 5

        self.state_encoder = RNNStateEncoder(
            512 * num_rnn_inputs, #TODO this might be too big, maybe combine visual encodings and pointnav encodings first
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

        self.device = None
        self.sdf_trainers = []
        self.config_file = "/home/karls/iSDF/isdf/train/configs/thor_live.json"

        self.train()



        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))



    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def init_new_map(self):
        sdf_map = trainer.Trainer(
            self.device,
            self.config_file,
            chkpt_load_file=None,
            incremental=True,
        )
        print("created new map")
        return sdf_map

    def format_frame_data(self, observations, timestep):
        data = FrameData(
            frame_id=torch.tensor([timestep]),
            im_batch=observations['rgb_lowres'],
            depth_batch=observations['depth_lowres'],
            T_WC_batch=torch.linalg.inv(observations['odometry_emul']['camera_info']['camera']['gt_transform'])
        )

        return data

    def update_maps(self, observations, masks):
        num_timesteps = observations['depth_lowres'].shape[0]
        num_batches = observations['depth_lowres'].shape[1]

        all_outputs = []
        for timestep in range(num_timesteps):
            batch_outputs = []
            for batch in range(num_batches):
                print("batch, {}/{} mask".format(batch, num_batches), masks[timestep][batch])
                if batch > len(self.sdf_trainers) - 1:
                    self.sdf_trainers.append(self.init_new_map())
                elif masks[timestep][batch] == 0:
                    print("resetting map")
                    self.sdf_trainers[batch] = self.init_new_map()
                
                frame_data = self.format_frame_data(observations, timestep)

                self.sdf_trainers[batch].add_frame(frame_data)
                if masks[timestep][batch] == 0:
                    self.sdf_trainers[batch].last_is_keyframe = True
                    self.sdf_trainers[batch].optim_frames = 200

                losses, step_time = self.sdf_trainers[batch].step()

                for i in range(self.sdf_trainers.optim_frames):
                    self.sdf_trainers[batch].step()

                output = self.sdf_trainers.compute_slices()
                batch_outputs.append(output)
                print("batch", batch, num_batches)
            batch_outputs = torch.stack(batch_outputs)
            all_outputs.append(batch_outputs)
        all_outputs = torch.stack(all_outputs)
        return all_outputs



    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        if self.device is None:
            self.device = self.full_visual_encoder_arm[0].bias.device


        all_maps = self.update_maps(observations, masks)
        print("all_maps", all_maps.shape)
        map_embedding = compute_cnn_output(self.map_encoder, all_maps)
        print("map_embedding", map_embedding.shape)

        #we really need to switch to resnet now that visual features are actually important
        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1

        agent_mask = observations['object_mask_source'].clone()
        agent_mask[after_pickup] = observations['object_mask_destination'][after_pickup]

        arm_mask = observations['object_mask_kinect_source'].clone()
        arm_mask[after_pickup] = observations['object_mask_kinect_destination'][after_pickup]

        #TODO remove these after issue is resolved
        observations['depth_lowres'] = check_for_nan_visual_observations(observations['depth_lowres'], where_it_occured='depth_lowres')
        observations['rgb_lowres'] = check_for_nan_visual_observations(observations['rgb_lowres'], where_it_occured='rgb_lowres')
        observations['depth_lowres_arm'] = check_for_nan_visual_observations(observations['depth_lowres_arm'], where_it_occured='depth_lowres_arm')
        observations['rgb_lowres_arm'] = check_for_nan_visual_observations(observations['rgb_lowres_arm'], where_it_occured='rgb_lowres_arm')
        arm_mask = check_for_nan_visual_observations(arm_mask, where_it_occured='arm_mask')
        agent_mask = check_for_nan_visual_observations(agent_mask, where_it_occured='agent_mask')

        visual_observation = torch.cat([observations['depth_lowres'], observations['rgb_lowres'], agent_mask], dim=-1).float()

        visual_observation_encoding_body = compute_cnn_output(self.full_visual_encoder, visual_observation)

        visual_observation_arm = torch.cat([observations['depth_lowres_arm'], observations['rgb_lowres_arm'], arm_mask], dim=-1).float()
        visual_observation_arm = check_for_nan_visual_observations(visual_observation_arm)
        visual_observation_encoding_arm = compute_cnn_output(self.full_visual_encoder_arm, visual_observation_arm)
        # ForkedPdb().set_trace()


        # arm_distance_to_obj_source = observations['point_nav_real_source'].copy()
        # arm_distance_to_obj_destination = observations['point_nav_real_destination'].copy()
        # arm_distance_to_obj_source_embedding = self.pointnav_embedding(arm_distance_to_obj_source)
        # arm_distance_to_obj_destination_embedding = self.pointnav_embedding(arm_distance_to_obj_destination)
        # pointnav_embedding = arm_distance_to_obj_source_embedding
        # pointnav_embedding[after_pickup] = arm_distance_to_obj_destination_embedding[after_pickup]

        #TODO remove
        observations['point_nav_emul_source'] = check_for_nan_visual_observations(observations['point_nav_emul_source'], where_it_occured='point_nav_emul_source')
        observations['point_nav_emul_destination'] = check_for_nan_visual_observations(observations['point_nav_emul_destination'], where_it_occured='point_nav_emul_destination')
        observations['arm_point_nav_emul_source'] = check_for_nan_visual_observations(observations['arm_point_nav_emul_source'], where_it_occured='arm_point_nav_emul_source')
        observations['arm_point_nav_emul_destination'] = check_for_nan_visual_observations(observations['arm_point_nav_emul_destination'], where_it_occured='arm_point_nav_emul_destination')

        agent_distance_to_obj_source = observations['point_nav_emul_source'].clone()
        agent_distance_to_obj_destination = observations['point_nav_emul_destination'].clone()
        #TODO eventually change this and the following to only calculate embedding for the ones we want
        agent_distance_to_obj_embedding_source = self.body_pointnav_embedding(agent_distance_to_obj_source)
        agent_distance_to_obj_embedding_destination = self.body_pointnav_embedding(agent_distance_to_obj_destination)
        agent_distance_to_obj_embedding = agent_distance_to_obj_embedding_source
        agent_distance_to_obj_embedding[after_pickup] = agent_distance_to_obj_embedding_destination[after_pickup]


        arm_distance_to_obj_source = observations['arm_point_nav_emul_source'].clone()
        arm_distance_to_obj_destination = observations['arm_point_nav_emul_destination'].clone()
        #TODO eventually change this and the following to only calculate embedding for the ones we want
        arm_distance_to_obj_embedding_source = self.arm_pointnav_embedding(arm_distance_to_obj_source)
        arm_distance_to_obj_embedding_destination = self.arm_pointnav_embedding(arm_distance_to_obj_destination)
        arm_distance_to_obj_embedding = arm_distance_to_obj_embedding_source
        arm_distance_to_obj_embedding[after_pickup] = arm_distance_to_obj_embedding_destination[after_pickup]


        if self.use_odom_pose:
            odom_observation = torch.stack([observations['odometry_emul']['agent_info']['xyz'][:, :, 0],
                                            observations['odometry_emul']['agent_info']['xyz'][:, :, 2],
                                            observations['odometry_emul']['agent_info']['rotation']], dim=-1).float()
            odom_embedding = self.odom_pose_embedding(odom_observation)

            visual_observation_encoding = torch.cat([visual_observation_encoding_body,
                                                     visual_observation_encoding_arm,
                                                     agent_distance_to_obj_embedding,
                                                     arm_distance_to_obj_embedding,
                                                     map_embedding,
                                                     odom_embedding], dim=-1)
        else:
            visual_observation_encoding = torch.cat([visual_observation_encoding_body,
                                                     visual_observation_encoding_arm,
                                                     agent_distance_to_obj_embedding,
                                                     arm_distance_to_obj_embedding,
                                                     map_embedding], dim=-1)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )


        # I think we need two model one for pick up and one for drop off

        actor_out_final = self.actor_pickup(x_out)
        critic_out_final = self.critic_pickup(x_out)

        actor_out_final = check_for_nan_visual_observations(actor_out_final, where_it_occured='actor_out_final')

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)


        if self.visualize:
            # if arm_distance_to_obj_source.shape[0] == 1 and arm_distance_to_obj_source.shape[1] == 1:
            #     arm_distances = arm_distance_to_obj_source
            #     arm_distances[after_pickup] = arm_distance_to_obj_destination
            #     agent_distances = agent_distance_to_obj_source
            #     agent_distances[after_pickup] = agent_distance_to_obj_destination
            #     print('arm_distances', arm_distances, 'agent_distances', agent_distances)

            source_object_mask = observations['object_mask_source']
            destination_object_mask = observations['object_mask_destination']
            intel_mask = source_object_mask
            intel_mask[after_pickup] = destination_object_mask[after_pickup]


            source_object_mask_kinect = observations['object_mask_kinect_source']
            destination_object_mask_kinect = observations['object_mask_kinect_destination']
            kinect_mask = source_object_mask_kinect
            kinect_mask[after_pickup] = destination_object_mask_kinect[after_pickup]

            distances = torch.cat([observations['point_nav_emul_source'],observations['arm_point_nav_emul_source']], dim=-1)
            # distances = observations['point_nav_emul_source']

            agent_distance_vector_to_viz =  observations['point_nav_emul_source'].clone()
            agent_distance_vector_to_viz[after_pickup] = observations['point_nav_emul_destination'][after_pickup].clone()

            arm_distance_vector_to_viz =  observations['arm_point_nav_emul_source'].clone()
            arm_distance_vector_to_viz[after_pickup] = observations['arm_point_nav_emul_destination'][after_pickup].clone()

            distance_vector_to_viz = dict(arm_dist=arm_distance_vector_to_viz, agent_dist=agent_distance_vector_to_viz)

            hacky_visualization(observations, 
                                object_mask=intel_mask, 
                                gt_mask=kinect_mask,
                                base_directory_to_right_images=self.starting_time,
                                text_to_write=distances,
                                distance_vector_to_viz=distance_vector_to_viz)




        return (
            actor_critic_output,
            memory,
        )

