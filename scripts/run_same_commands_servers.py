import argparse
import os
import pdb

command_aws1 = ''
command_aws5 = ''
command_aws15 = ''
command_vs411 = ''
command_aws8 = ''
command_aws12 = ''

# command_aws8 = ' source ~/manipulathor_env/bin/activate; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/objectnav/only_explore_for_all_robothor_room_rgb_only_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#    --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#    --seed 10 --machine_id 0 --extra_tag only_explore_for_all_robothor_room_rgb_only '

BASE_COMMAND = ' source ~/manipulathor_env/bin/activate; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact EXPERIMENT_CONFIG \
  --distributed_ip_and_port MAIN_IP:6060 \
   --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
   --seed 10 --machine_id 0 --extra_tag EXPERIMENT_NAME '


server_mapping = dict(
    aws1 = {
        'servers':[f'aws{i}' for i in range(1,5)],
    },
    aws5 = {
        'servers':[f'aws{i}' for i in range(5, 8)],
    },
    aws8 = {
        'servers':[f'aws{i}' for i in range(8, 12)],
    },
    aws12 = {
        'servers':[f'aws{i}' for i in range(12,16)],
    },
    aws15 = {
        'servers':[f'aws{i}' for i in range(1, 9)],
    },

    vs411 = {
        'servers':['vision-server11', 'vision-server4'],
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--server_set', default=None)
    parser.add_argument('--command', default=BASE_COMMAND, type=str)
    parser.add_argument('--directly', action='store_true')
    parser.add_argument('--experiment_config', type=str, required=True)
    parser.add_argument('--weight_adr_on_main_server', default=None, type=str)
    parser.add_argument('--experiment_name', default=None, type=str)

    args = parser.parse_args()

    args.servers = []
    info_for_server = server_mapping[args.server_set]
    args.servers = info_for_server['servers']


    args.command = args.command.replace('NUM_MACHINES', str(len(args.servers)))
    args.command = args.command.replace('EXPERIMENT_CONFIG', args.experiment_config)
    if args.weight_adr_on_main_server is not None:
        args.command = f'scp MAIN_SERVER:{args.weight_adr_on_main_server} ~/;' + args.command
        weight_name = args.weight_adr_on_main_server.split('/')[-1]
        args.command += f' -c ~/{weight_name}'
    if args.experiment_name is None:
        args.experiment_name = args.experiment_config.split('/')[-1]
    args.command = args.command.replace('EXPERIMENT_NAME', args.experiment_name)
    main_server_ip_adr_w_username = get_ip_adr_from_config(args.servers[0])
    args.command = args.command.replace('MAIN_SERVER', main_server_ip_adr_w_username)
    args.command = args.command.replace('MAIN_IP', get_ip_adr_from_config(args.servers[0], just_ip=True))


    return args

def get_ip_adr_from_config(server_name, just_ip=False):
    with open('scripts/config') as f:
        all_lines = [l for l in f]
        for i in range(len(all_lines)):
            if all_lines[i] == f'Host {server_name}\n':
                username = all_lines[i + 2]
                assert 'User' in username
                username = username.split(' ')[-1].replace('\n', '')
                ip_adr = all_lines[i + 1]
                assert 'HostName' in ip_adr
                ip_adr = ip_adr.split(' ')[-1].replace('\n', '')
                if just_ip:
                    return ip_adr
                return f'{username}@{ip_adr}'
    raise Exception('Ip not found', server_name)


def main(args):

    for (i, server) in enumerate(args.servers):
        # server_id = int(server.replace('aws', '')) - 1
        command_to_run = args.command.replace('--machine_id 0', f'--machine_id {i}')
        print('command to run', command_to_run)
        os.system(f'echo \"{command_to_run}\" > ~/command_to_run.sh')
        server = get_ip_adr_from_config(server)
        os.system(f'rsync ~/command_to_run.sh {server}:~/')
        os.system(f'ssh {server} chmod +x command_to_run.sh')

        if args.directly:
            command = f'ssh {server} ./command_to_run.sh&'
            os.system(command)



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


