import os
import pdb
import time
import psutil

servers = [('aws1', 6006), ('aws5', 6007)]


while(True):

    for p in psutil.process_iter():
        if 'ssh' in p.name() and '-NfL' in p.cmdline():

            p.terminate()
            print('killed', p.pid)
    for servername, port in servers:
        command = f'ssh -NfL {port}:localhost:{port} {servername}'
        os.system(command)

    pdb.set_trace()
