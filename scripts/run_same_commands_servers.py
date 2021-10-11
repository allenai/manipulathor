import argparse
import os
import pdb

# command = './manipulathor/scripts/kill-zombie.sh'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
#   --distributed_ip_and_port 34.220.30.46:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0'
command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_memory_distrib \
  --distributed_ip_and_port 34.220.30.46:6060 \
  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
  --seed 10 --machine_id 0 -c ~/exp_ComplexRewardNoPUWMemory__stage_00__steps_000045112992.pt'
# command = 'scp ec2-34-220-30-46.us-west-2.compute.amazonaws.com:~/manipulathor/experiment_output/checkpoints/ComplexRewardNoPUWMemory/2021-10-08_23-12-59/exp_ComplexRewardNoPUWMemory__stage_00__steps_000045112992.pt ~/'
list_of_servers = ['aws1', 'aws2', 'aws3', 'aws4', ]

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--servers', default=None, nargs='+')
    parser.add_argument('--command', default=command, type=str)
    parser.add_argument('--directly', action='store_true')

    args = parser.parse_args()
    if args.servers is None:
        args.servers = list_of_servers
    else:
        args.servers = args.servers
    return args

def main(args):

    for server in args.servers:
        if args.directly:
            command = f'ssh {server} {args.command}'
            print('executing', command)
            os.system(command)
            print('done')
        else:
            server_id = int(server.replace('aws', '')) - 1
            command_to_run = args.command.replace('--machine_id 0', f'--machine_id {server_id}')
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


