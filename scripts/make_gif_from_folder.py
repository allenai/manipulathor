import argparse
import os
import pdb

import moviepy.video.io.ImageSequenceClip


from ithor_arm.ithor_arm_viz import save_image_list_to_gif
import cv2
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('-f', '--folder_name', default=None)
    parser.add_argument('--output_type', default='gif')
    parser.add_argument('--max_len', default=-1, type=int)


    args = parser.parse_args()
    return args

args = parse_args()
folder_name = args.folder_name
image_names = [f for f in glob.glob(os.path.join(folder_name, '*.png'))]
image_names = sorted(image_names)

if args.max_len > 0:
    image_names = image_names[:args.max_len]


if args.output_type == 'gif':
    all_images = []
    for img_name in image_names:
        im = cv2.imread(img_name)[:, :, [2,1,0]]
        all_images.append(im)
    concat_all_images = np.expand_dims(np.stack(all_images, axis=0), axis=1)
    save_image_list_to_gif(concat_all_images, 'generated_gif.gif', folder_name)
elif args.output_type == 'mp4':
    print('reading images')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_names, fps=3)
    print('making the video')
    clip.write_videofile(os.path.join(folder_name, 'generated_video.mp4'))