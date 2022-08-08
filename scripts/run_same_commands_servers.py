import argparse
import os
import pdb

command_aws1 = ''
command_aws5 = ''
command_aws15 = ''
command_vs411 = ''
command_aws8 = ''
command_aws12 = ''

command_aws8 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjExploreITHORAllRoomsRGBOnlyDistrib/object_explore_with_clip/2022-08-05_01-10-17/exp_ObjExploreITHORAllRoomsRGBOnlyDistrib_object_explore_with_clip__stage_00__steps_000017614273.pt ~/; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/objectnav/object_explore_for_ithor_rgb_only_distrib \
 --distributed_ip_and_port IP_ADR:6060 \
  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
  --seed 10 --machine_id 0 --extra_tag object_explore_with_clip -c ~/exp_ObjExploreITHORAllRoomsRGBOnlyDistrib_object_explore_with_clip__stage_00__steps_000017614273.pt'

command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjNavSmallSetITHORAllRoomsRGBOnlyDistrib/object_nav_small_source_with_clip/2022-08-07_05-05-08/exp_ObjNavSmallSetITHORAllRoomsRGBOnlyDistrib_object_nav_small_source_with_clip__stage_00__steps_000009087399.pt ~/; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/objectnav/object_nav_w_smaller_set_for_ithor_rgb_only_distrib \
 --distributed_ip_and_port IP_ADR:6060 \
  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
  --seed 10 --machine_id 0 --extra_tag object_nav_small_source_with_clip -c ~/exp_ObjNavSmallSetITHORAllRoomsRGBOnlyDistrib_object_nav_small_source_with_clip__stage_00__steps_000009087399.pt'


server_mapping = dict(
    aws1 = {
        'servers':[f'aws{i}' for i in range(1,5)],
        'ip_adr': '18.237.24.199',
        'command': command_aws1,
    },
    aws5 = {
        'servers':[f'aws{i}' for i in range(5, 8)],
        'ip_adr': '34.216.219.227',
        'command': command_aws5,
    },
    aws8 = {
        'servers':[f'aws{i}' for i in range(8, 12)],
        'ip_adr': '35.90.135.47',
        'command': command_aws8,
    },
    aws12 = {
        'servers':[f'aws{i}' for i in range(12,16)],
        'ip_adr': '35.91.24.190',
        'command': command_aws12,
    },
    aws15 = {
        'servers':[f'aws{i}' for i in range(1, 9)],
        'ip_adr': '18.237.24.199',
        'command': command_aws15,
    },

    vs411 = {
        'servers':['vision-server11', 'vision-server4'],
        'ip_adr': '172.16.122.186',
        'command': command_vs411,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--server_set', default=None)#, nargs='+')
    parser.add_argument('--command', default=None, type=str)
    parser.add_argument('--directly', action='store_true')

    args = parser.parse_args()

    args.servers = []
    info_for_server = server_mapping[args.server_set]
    args.servers = info_for_server['servers']
    ip_adr = info_for_server['ip_adr']
    if args.command is None:
        args.command = info_for_server['command']

    args.command = args.command.replace('MAIN_SERVER', ip_adr)
    args.command = args.command.replace('IP_ADR', ip_adr)
    args.command = args.command.replace('NUM_MACHINES', str(len(args.servers)))
    return args

def main(args):

    for (i, server) in enumerate(args.servers):
        if args.directly:
            command = f'ssh {server} {args.command}'
            print('executing', command)
            os.system(command)
            print('done')
        else:
            # server_id = int(server.replace('aws', '')) - 1
            command_to_run = args.command.replace('--machine_id 0', f'--machine_id {i}')
            print('command to run', command_to_run)
            os.system(f'echo \"{command_to_run}\" > ~/command_to_run.sh')
            os.system(f'rsync ~/command_to_run.sh {server}:~/')
            os.system(f'ssh {server} chmod +x command_to_run.sh')
            command = f'ssh {server} ./command_to_run.sh'



if __name__ == '__main__':
    args = parse_args()
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


