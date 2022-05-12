import os
import pdb
import time
import psutil

servers = [('aws1', 6006), ('aws5', 6007)]


while(True):

    for p in psutil.process_iter():
        try:
            if 'ssh' in p.name() and '-NfL' in p.cmdline():
                p.terminate()
                print('killed', p.pid)
        except Exception:
            continue
    for servername, port in servers:
        command = f'ssh -NfL {port}:localhost:{port} {servername}'
        os.system(command)

    pdb.set_trace()
