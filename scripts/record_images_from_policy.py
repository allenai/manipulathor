import argparse
import numpy as np
import imageio

import json
from os.path import join
import random
import gym
from tqdm import tqdm
from manipulathor_baselines.stretch_bring_object_baselines.experiments.ithor.pointnav_emul_stretch_all_rooms_with_map import PointNavEmulStretchAllRooms

def denormalize_image(image):
    denormed = image * np.array([0.229, 0.224, 0.225])
    denormed = denormed + np.array([[0.485, 0.456, 0.406]])
    denormed = np.clip(denormed, 0.0, 1.0)
    denormed = denormed * 255
    denormed = denormed.astype(np.uint8)
    return denormed

def depth_to_uint(depth, max_depth=5.0):
    converted = depth / max_depth * 65536
    converted = converted.astype(np.uint16)
    return converted


def save_images(obs, out_path, episode, index):
    im = denormalize_image(obs['rgb_lowres'])
    imageio.imwrite(join(out_path, 'rgb_episode{:05d}_step{:05d}.jpg'.format(episode, index)), im)
    im_arm = denormalize_image(obs['rgb_lowres_arm'])
    imageio.imwrite(join(out_path,'rgb_arm_episode{:05d}_step{:05d}.jpg'.format(episode, index)), im_arm)

    depth = depth_to_uint(obs['depth_lowres'])
    imageio.imwrite(join(out_path, 'depth_episode{:05d}_step{:05d}.png'.format(episode, index)), depth)
    depth_arm = depth_to_uint(obs['depth_lowres_arm'])
    imageio.imwrite(join(out_path, 'depth_arm_episode{:05d}_step{:05d}.png'.format(episode, index)), depth_arm)

def record_images_from_policy(args):
    config = PointNavEmulStretchAllRooms()
    config.ENV_ARGS['motion_noise_type'] = 'habitat'
    config.ENV_ARGS['motion_noise_args'] = dict()
    config.ENV_ARGS['motion_noise_args']['multiplier_means'] = [1,1,1,1,1,1]
    config.ENV_ARGS['motion_noise_args']['multiplier_sigmas'] = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    config.ENV_ARGS['motion_noise_args']['effect_scale'] = .25

    mode = "val" if args.valid else "train"

    task_sampler = PointNavEmulStretchAllRooms.make_sampler_fn(scenes=config.TRAIN_SCENES,
                                                            sensors=config.SENSORS,
                                                            max_steps=config.MAX_STEPS,
                                                            env_args=config.ENV_ARGS,
                                                            action_space=gym.spaces.Discrete(len(config.TASK_TYPE.class_action_names())),
                                                            rewards_config=config.REWARD_CONFIG,
                                                            sampler_mode=mode,
                                                            cap_training=config.CAP_TRAINING)
    task_sampler.visualizers = []

    pairs_name = "pairs.json" if args.start_index == 0 else "pairs_{}.json".format(args.start_index)

    action_results = {}
    dataset = []
    for episode_index in tqdm(range(args.episodes_to_save)):
        episode = episode_index + args.start_index
        task = task_sampler.next_task()

        index = 0
        obs = task.get_observations()
        save_images(obs, args.out_path, episode, index)
        
        index += 1
        while not task.is_done():
            actions = task.class_action_names()
            # motion_indices = [i for i in range(len(actions)) if actions[i] in ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]]
            if np.random.rand() < 0.75:
                motion_indices = [i for i in range(len(actions)) if actions[i] in ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight", "RotateLeftSmall", "RotateRightSmall"]]
                action = random.choice(motion_indices)
            else:
                action = np.random.randint(len(actions))
            # action = 9
            # print(actions[action], action)

            obs = task.step(action).observation
            if actions[action] not in action_results.keys():
                action_results[actions[action]] = []
            action_results[actions[action]].append(obs['odometry_emul']['agent_info']['relative_xyz'])

            save_images(obs, args.out_path, episode, index)
            pair = {
                'episode': episode,
                'step': index,
                'action': actions[action],
                'noisy_rotation': obs['odometry_emul']['agent_info']['noisy_relative_rot'],
                'noisy_translation': obs['odometry_emul']['agent_info']['noisy_relative_xyz'].tolist(),
                'gt_rotation':  obs['odometry_emul']['agent_info']['relative_rot'],
                'gt_translation': obs['odometry_emul']['agent_info']['relative_xyz'].tolist(),
            }
            dataset.append(pair)

            index += 1

        with open(join(args.out_path, pairs_name), 'w') as f:
            json.dump(dataset, f, indent=4)

        # for k in action_results:
        #     if 'Move' not in k:
        #         continue
        #     print("\n",k)
        #     for i in action_results[k]:
        #         print(i)
        task.finish_visualizer(True)

    task_sampler.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("--valid", dest="valid", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--episodes_to_save", type=int, default=10)
    args = parser.parse_args()

    record_images_from_policy(args)
