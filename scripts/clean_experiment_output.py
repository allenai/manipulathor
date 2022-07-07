import argparse
import os
from os.path import isfile, isdir, join
from os import listdir
import shutil

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def rm_if_dir(path):
    if isdir(path):
        shutil.rmtree(path)

def clean_dir(path, min_checkpoint_size, min_tb_size, delete=False, list_all=False):
    tb_path = join(path, "tb")
    ckpt_path = join(path, "checkpoints")
    cfg_path = join(path, "used_configs")
    experiments = [d for d in listdir(cfg_path) if isdir(join(cfg_path, d))]

    print("TB Size\tCkpt size \tName")
    for experiment in experiments:
        seeds = [d for d in listdir(join(cfg_path, experiment)) if isdir(join(cfg_path, experiment, d))]

        for seed in seeds:
            tb_size = get_size(join(tb_path, experiment, seed))

            
            ckpt_size = get_size(join(ckpt_path, experiment, seed))
            
            if list_all:
                print(tb_size, "\t", ckpt_size, "\t", seed)
                continue
            if tb_size > min_tb_size:
                continue
            if ckpt_size > min_checkpoint_size:
                continue
            print(tb_size, "\t", ckpt_size, "\t", seed)

            if delete:
                rm_if_dir(join(tb_path, experiment, seed))
                rm_if_dir(join(ckpt_path, experiment, seed))
                rm_if_dir(join(cfg_path, experiment, seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--min_checkpoint_size', type=int, default=-1, 
                        help="Deletes all experiments with a checkpoint folder less than n bytes")
    parser.add_argument('--min_tb_size', type=int, default=-1,
                        help="Deletes all experiments with a tb folder less than n bytes")
    parser.add_argument("--list_all", default=False, action='store_true',
                        help='Lists the sizes and names of all the experiments')
    parser.add_argument('--delete', default=False, action='store_true',
                        help='Use this flag to actually delete the detected folders.')
    args = parser.parse_args()
    clean_dir(args.path, args.min_checkpoint_size, args.min_tb_size, args.delete, args.list_all)