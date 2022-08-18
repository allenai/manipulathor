import argparse
import glob
import pdb
import os
import time

import psutil

from scripts.run_same_commands_servers import get_ip_adr_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--server_set', default=None, required=True)
    return parser.parse_args()


def restart_experiment(server_set):

    #find the latest weight that exists
    base_dir = 'experiment_output/checkpoints/'
    files = glob.glob(os.path.join(base_dir, '**', '*.pt'), recursive=True)
    latest_file = max(files, key=os.path.getctime)
    full_adr_of_latest_weight = os.path.abspath(latest_file)

    home_directory = (os.path.expanduser('~'))
    #load the latest experiment_config
    with open(os.path.join(home_directory, 'command_to_run.sh')) as f:
        latest_command = [l for l in f]
        assert len(latest_command) == 1
        latest_command = latest_command[0]


    last_experiment_config = None
    main_server_adr = None
    splitted_command = latest_command.split(' ')
    for i in range(len(splitted_command)):
        if splitted_command[i] == 'allenact':
            last_experiment_config = splitted_command[i + 1]
        if splitted_command[i] == '--distributed_ip_and_port':
            main_server_adr = splitted_command[i + 1].split(':')[0]

    assert last_experiment_config and main_server_adr


    this_server_ip = get_ip_adr_from_config(server_set, just_ip=True)
    assert main_server_adr == this_server_ip # To make sure the main server is the one we are running this command from

    #add the copying of the weight from the main server
    #replace the previous weight with the new weight
    #rewrite the command on every server
    #run the command on every server
    command_to_sync = f'python3 scripts/run_same_commands_servers.py --server_set {server_set} --experiment_config {last_experiment_config} --weight_adr_on_main_server {full_adr_of_latest_weight} --directly'
    os.system(command_to_sync)

if __name__ == '__main__':
    args = parse_args()
    server_set = args.server_set
    while(True):
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load15/os.cpu_count()) * 100
        print(cpu_usage)
        time.sleep(2) #TODO_NOW this needs to be less frequent
        if cpu_usage < 28: #TODO_NOW change to 10
            restart_experiment(server_set)
            print('experiments were restarted')
            break