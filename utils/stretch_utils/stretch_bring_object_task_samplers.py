"""Task Samplers for the task of ArmPointNav"""
import json
import random

from torch.distributions.utils import lazy_property

from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from manipulathor_utils.debugger_util import ForkedPdb
from utils.manipulathor_data_loader_utils import get_random_query_image, get_random_query_feature_from_img_adr
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from scripts.stretch_jupyter_helper import get_reachable_positions, transport_wrapper
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer


class StretchDiverseBringObjectTaskSampler(DiverseBringObjectTaskSampler):
    def _create_environment(self, **kwargs) -> StretchManipulaTHOREnvironment:
        env = StretchManipulaTHOREnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env
    def reset_scene(self, scene_name):
        self.env.reset(
            scene_name=scene_name, agentMode="stretch", agentControllerType="mid-level"
        )
    def next_task(
            self, force_advance_scene: bool = False
    ) :

        #TODO this is too entangled, double check that we have redefined all the functions we use

        if self.env is None:
            self.env = self._create_environment()

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        data_point = self.get_source_target_indices()

        scene_name = data_point["scene_name"]
        init_object = data_point['init_object']
        goal_object = data_point['goal_object']
        agent_state = data_point["initial_agent_pose"]

        assert init_object["scene_name"] == goal_object["scene_name"] == scene_name
        assert init_object['object_id'] != goal_object['object_id']

        self.reset_scene(scene_name)

        this_controller = self.env



        def put_object_in_location(location_point):

            object_id = location_point['object_id']
            location = location_point['object_location']
            event, _ = transport_wrapper(this_controller,object_id,location,)
            return event

        event_transport_init_obj = put_object_in_location(init_object)
        event_transport_goal_obj = put_object_in_location(goal_object)

        if not event_transport_goal_obj.metadata['lastActionSuccess'] or not event_transport_init_obj.metadata['lastActionSuccess']:
            print('scene', scene_name, 'init', init_object['object_id'], 'goal', goal_object['object_id'])
            print('ERROR: one of transfers fail', 'init', event_transport_init_obj.metadata['errorMessage'], 'goal', event_transport_goal_obj.metadata['errorMessage'])

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        if not event.metadata['lastActionSuccess']:
            print('ERROR: Teleport failed')


        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), StretchBringObjImageVisualizer)
        ]

        initial_object_info = self.env.get_object_by_id(init_object["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()

        # source_img_adr = get_random_query_image_file_name(scene_name,init_object['object_id'], self.query_image_dict)
        # goal_img_adr = get_random_query_image_file_name(scene_name,goal_object['object_id'], self.query_image_dict)

        source_object_query, source_img_adr = get_random_query_image(scene_name,init_object['object_id'], self.query_image_dict)
        goal_object_query, goal_img_adr = get_random_query_image(scene_name,goal_object['object_id'], self.query_image_dict)
        source_object_query_feature = get_random_query_feature_from_img_adr(source_img_adr)
        goal_object_query_feature = get_random_query_feature_from_img_adr(goal_img_adr)


        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            "init_location": init_object,
            "goal_location": goal_object,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
            'source_object_query': source_object_query,
            'goal_object_query': goal_object_query,
            'source_object_query_feature': source_object_query_feature,
            'goal_object_query_feature': goal_object_query_feature,
            # 'source_object_query_file_name' : source_img_adr,
            # 'goal_object_query_file_name' : goal_img_adr,
            'episode_number': random.uniform(0, 10000),
        }


        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = init_object
            task_info["visualization_target"] = goal_object

        self._last_sampled_task = self.TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    @lazy_property
    def stretch_reachable_positions(self):
        with open('datasets/apnd-dataset/stretch_init_location.json') as f:
            return json.load(f)

    def get_source_target_indices(self):
        data_point = super().get_source_target_indices()

        #TODO this needs to be fixed for test
        reachable_positions = self.stretch_reachable_positions[data_point['scene_name']]
        agent_pose = random.choice(reachable_positions)



        data_point['initial_agent_pose'] = {
            "name": "agent",
            "position": dict(x=agent_pose['x'], y=agent_pose['y'], z=agent_pose['z']),
            "rotation": dict(x=0, y=agent_pose['rotation'], z=0),
            "cameraHorizon": data_point['initial_agent_pose']['cameraHorizon'],
            "isStanding": True,
        }

        return data_point
