"""Utility functions and classes for visualization and logging"""
import os
from datetime import datetime
import numpy as np

import imageio

from plugins.ithor_arm_plugin.arm_calculation_utils import initialize_arm
from plugins.ithor_arm_plugin.ithor_arm_constants import reset_environment_and_additional_commands, transport_wrapper
from utils.debugger_util import ForkedPdb
import cv2

class LoggerVisualizer:
    def __init__(self, exp_name='', log_dir=''):
        if log_dir == '':
            log_dir = self.__class__.__name__
        if exp_name == '':
            exp_name = 'NoNameExp'
        now = datetime.now()
        self.exp_name = exp_name
        log_dir = os.path.join('experiment_output/visualizations', exp_name, log_dir + '_' + now.strftime("%m_%d_%Y_%H_%M_%S_%f"))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_queue = []
        self.action_queue = []
        self.logger_index = 0

    def log(self, environment, action_str):
        raise Exception("Not Implemented")


    def is_empty(self):
        return len(self.log_queue) == 0

    def finish_episode_metrics(self, episode_info, task_info, metric_results):
        pass

    def finish_episode(self, environment, episode_info, task_info):
        pass


class TestMetricLogger(LoggerVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_metric_dict = {}
        log_file_name = os.path.join(self.log_dir, 'test_metric_{}.txt'.format(self.exp_name))
        self.metric_log_file = open(log_file_name, 'w')

    def average_dict(self):
        result = {}
        for (k, v) in self.total_metric_dict.items():
            result[k] = sum(v) / len(v)
        return result
    def finish_episode_metrics(self, episode_info, task_info, metric_results=None):

        if metric_results is None:
            print('had to reset')
            self.log_queue = []
            self.action_queue = []
            return

        for k in metric_results.keys():
            if 'metric' in k or k in ['ep_length', 'reward', 'success']:
                self.total_metric_dict.setdefault(k, [])
                self.total_metric_dict[k].append(metric_results[k])
        print('total', len(self.total_metric_dict['success']),'average test metric', self.average_dict())


        # save the task info and all the action queue and results
        log_dict = {
            'task_info_metrics': metric_results,
            'action_sequence': self.action_queue,
            'logger_number': self.logger_index,
        }
        self.logger_index += 1
        self.metric_log_file.write(str(log_dict)); self.metric_log_file.write('\n')
        print('Logging to', self.metric_log_file.name)

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        # We can add agent arm and state location if needed
        self.action_queue.append(action_str)
        self.log_queue.append(action_str)

class ImageVisualizer(LoggerVisualizer):

    def finish_episode(self, environment, episode_info, task_info):
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        time_to_write += 'log_ind_{}'.format(self.logger_index)
        self.logger_index += 1
        print('Loggigng', time_to_write, 'len', len(self.log_queue))
        object_id = task_info['objectId']

        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        #Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = 'succ' if episode_success else 'fail'
        pickup_success_offset = 'succ' if pickup_success else 'fail'
        gif_name = (time_to_write + '_obj_' + object_id.split('|')[0] + '_pickup_' + pickup_success_offset + '_episode_' + episode_success_offset + '.gif')
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0),axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)

        self.log_start_goal(environment, task_info['visualization_source'], tag='start', img_adr = os.path.join(self.log_dir, time_to_write ))
        self.log_start_goal(environment, task_info['visualization_target'], tag='goal', img_adr = os.path.join(self.log_dir, time_to_write ))

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        image_tensor = environment.current_frame
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)

    def log_start_goal(self, env, task_info, tag, img_adr):
        object_location = task_info['object_location']
        object_id = task_info['object_id']
        agent_state = task_info['agent_pose']
        this_controller = env.controller
        scene = this_controller.last_event.metadata['sceneName'] # maybe we need to reset env actually]
        reset_environment_and_additional_commands(this_controller, scene)
        # event = this_controller.step(dict(action = 'DropMidLevelHand'))
        # for start arm from high up as a cheating, this block is very important. never remove
        event1, event2, event3 = initialize_arm(this_controller)
        if not(event1.metadata['lastActionSuccess'] and event2.metadata['lastActionSuccess'] and event3.metadata['lastActionSuccess']):
            print('ERROR: ARM MOVEMENT FAILED in logging! SHOULD NEVER HAPPEN')

        event = transport_wrapper(this_controller, object_id, object_location)
        if event.metadata['lastActionSuccess'] == False:
            print('ERROR: oh no could not transport in logging')


        # initial_pose = scene_start_cheating_init_pose[scene]
        # event1 = this_controller.step(dict(action='TeleportFull', x=initial_pose['x'], y=initial_pose['y'], z=initial_pose['z'], rotation=dict(x=0, y=initial_pose['rotation'], z=0), horizon=initial_pose['horizon']))
        # this_controller.step(dict(action='PausePhysicsAutoSim'))
        # event2 = this_controller.step(dict(action='MoveMidLevelArm',  position=dict(x=0.0, y=0, z=0.35), **ADITIONAL_ARM_ARGS))
        # event3 = this_controller.step(dict(action='MoveMidLevelArmHeight', y=0.8, **ADITIONAL_ARM_ARGS))



            # print('scene', scene, initial_pose, ADITIONAL_ARM_ARGS)
            # print(event1.metadata['actionReturn'] , event2.metadata['actionReturn'] , event3.metadata['actionReturn'])


        event = this_controller.step(dict(action='TeleportFull', standing=True, x=agent_state['position']['x'], y=agent_state['position']['y'], z=agent_state['position']['z'], rotation=dict(x=agent_state['rotation']['x'], y=agent_state['rotation']['y'], z=agent_state['rotation']['z']), horizon=agent_state['cameraHorizon']))
        if event.metadata['lastActionSuccess'] == False:
            print('ERROR: oh no could not teleport in logging')

        image_tensor = this_controller.last_event.frame
        image_dir = img_adr + '_obj_' + object_id.split('|')[0] + '_pickup_' + tag + '.png'
        cv2.imwrite(image_dir, image_tensor[:,:,[2,1,0]])



class ThirdViewVisualizer(LoggerVisualizer):

    def __init__(self):
        super(ThirdViewVisualizer, self).__init__()
        print('This does not work')
        ForkedPdb().set_trace()
        # self.init_camera = False


    def finish_episode(self, environment, episode_success, task_info):
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        print('Loggigng', time_to_write, 'len', len(self.log_queue))

        for i, img in enumerate(self.log_queue):
            image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
            cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        success_offset = 'succ' if episode_success else 'fail'
        gif_name = (time_to_write + '_' + success_offset + '.gif')
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0),axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):

        # if True or not self.init_camera:
        #     # self.init_camera = True
        #
        #     agent_state = environment.controller.last_event.metadata['agent']
        #     offset={'x':1, 'y':1, 'z':1}
        #     rotation_offset = 0
        #     # environment.controller.step('UpdateThirdPartyCamera', thirdPartyCameraId=0, rotation=dict(x=0, y=agent_state['rotation']['y']+rotation_offset, z=0), position=dict(x=agent_state['position']['x'] + offset['x'], y=1.0 + offset['y'], z=agent_state['position']['z'] + offset['z']))
        #     environment.controller.step('UpdateThirdPartyCamera', thirdPartyCameraId=0, rotation=dict(x=0, y=45, z=0), position=dict(x=-1, y=1.5, z=-1), fieldOfView=100)


        # the direction of this might not be ideal
        image_tensor = environment.controller.last_event.third_party_camera_frames[0]
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)


# def __save_thirdparty_camera(controller, address='/Users/kianae/Desktop/third_camera.png', offset={'x':0, 'y':0, 'z':0}, rotation_offset=0):
#     This is taking an additional step which messes up the sequence
#     agent_state = controller.last_event.metadata['agent']
#     controller.step('UpdateThirdPartyCamera', thirdPartyCameraId=0, rotation=dict(x=0, y=agent_state['rotation']['y']+rotation_offset, z=0), position=dict(x=agent_state['position']['x'] + offset['x'], y=1.0 + offset['y'], z=agent_state['position']['z'] + offset['z']))
#     frame = controller.last_event.third_party_camera_frames
#     assert len(frame) == 1
#     frame = frame[0]
#     file_adr = address
#     res = cv2.imwrite(file_adr, frame[:,:,[2,1,0]])
#     return res

def save_image_list_to_gif(image_list, gif_name, gif_dir):
    gif_adr = os.path.join(gif_dir, gif_name)

    seq_len, cols, w, h, c = image_list.shape

    pallet = np.zeros((seq_len, w, h * cols, c))

    for col_ind in range(cols):
        pallet[:, :, col_ind * h: (col_ind + 1) * h, :] = image_list[:, col_ind]

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, pallet.astype(np.uint8), format='GIF', duration=1 / 5)
    print('Saved result in ', gif_adr)

