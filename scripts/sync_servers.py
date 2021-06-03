import argparse
import os
import pdb

list_of_servers = ['kiana-workstation', 'vision-server12', 'aws15', 'aws16','aws14','aws17', 'vision-server13']

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--servers', default=None)
    parser.add_argument('--sync_weights', default=False, action='store_true')
    parser.add_argument('--sync_specific_weight', default=None)
    parser.add_argument('--sync_back', default=False, action='store_true')


    args = parser.parse_args()
    if args.servers is None:
        args.servers = list_of_servers
    else:
        args.servers = [args.servers]
    return args

def main(args):

    for server in args.servers:
        print('syncing to ', server)
        command = 'rsync  -avz --copy-links\
             --exclude .idea \
             --exclude __pycache__/ \
             --exclude runs/ \
             --exclude .DS_Store \
             --exclude .direnv \
             --exclude .envrc \
             --exclude .git \
             --exclude experiment_output/ \
             --exclude test_out/ \
             --exclude docs/ \
             --exclude trained_weights/  \
             --exclude pretrained_models/  \
             --exclude trained_weights/do_not_sync_weights/  \
             ../manipulathor {}:~/'.format(server)
        if args.sync_weights:
            command = command.replace('--exclude trained_weights/ ', '')
        if args.sync_specific_weight is not None:
            print('Not implemented yet')
            pdb.set_trace()
            command = command.replace('--exclude trained_weights/ ', '')
        os.system(command)

def sync_back(args):
    for server in args.servers:
        print('syncing from ', server)
        command = 'rsync  -avz \
             --exclude *.png \
             --exclude *.gif \
             {}:~/allenact/experiment_output/visualizations ~/Desktop/server_results_sync/{}'.format(server, server)

        # rsync -avz --exclude "*.png" vision-server12:~/allenact/experiment_output/visualizations/RealDepthRandomAgentLocArmNav_03_24_2021_17_46_17_098605/ depth_visualizations

        if args.sync_weights:
            command = command.replace('--exclude ImageVisualizer/ ', '')
        os.system(command)

if __name__ == '__main__':
    args = parse_args()
    if args.sync_back:
        sync_back(args)
    else:
        main(args)


# On each server:
# echo "set-option -g history-limit 3000000" >> ~/.tmux.conf
# tmux new
# sudo apt-get install xinit
# sudo python3 scripts/startx.py&
# #sudo apt-get install python3-dev
# #sudo apt-get install libffi-dev
# sudo pip3.6 install -r edited_small_requirements.txt
# #sudo mount /dev/sda1 ~/storage
# git init; git remote add origin https://github.com/ehsanik/dummy.git; git add README.md; git commit -am "something"
# sudo pip3.6 install -e git+https://github.com/allenai/ai2thor.git@43f62a0aa2a1aaafb6fd05d28bea74fdc866eea1#egg=ai2thor
# python3.6 >>>> import ai2thor.controller; c=ai2thor.controller.Controller(); c._build.url
# tensorboard --logdir experiment_output/tb --bind_all --port
# python3.6 main.py -o experiment_output -b projects/armnav_baselines/experiments/ithor/ armnav_ithor_rgb_simplegru_ddppo
# #ssh -NfL 6015:localhost:6015 aws15;ssh -NfL 6016:localhost:6016 aws16;ssh -NfL 6014:localhost:6014 aws14;ssh -NfL 6017:localhost:6017 aws17;
