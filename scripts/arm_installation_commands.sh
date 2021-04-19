#!/bin/zsh
# Use deep learning instance, with g4dn.12xlarge, change storage
#curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#sudo apt-get install python3-distutils
#sudo apt-get install python3-apt
#sudo python3 get-pip.py
#deep learning 18.04, 41
#g4dn.12xlarge
#500G Storage
#Name

cd ~/allenact
echo "set-option -g history-limit 3000000" >> ~/.tmux.conf
tmux new
sudo apt-get install xinit
sudo python3 scripts/startx.py&
#sudo apt-get install python3-dev
#sudo apt-get install libffi-dev
sudo pip3.6 install -r edited_small_requirements.txt
#sudo mount /dev/sda1 ~/storage
git init
git remote add origin https://github.com/ehsanik/dummy.git
git add README.md
git commit -am "something"
sudo pip3.6 install -e git+https://github.com/allenai/ai2thor.git@43f62a0aa2a1aaafb6fd05d28bea74fdc866eea1#egg=ai2thor
python3.6 >>>> import ai2thor.controller; c=ai2thor.controller.Controller(); c._build.url
tensorboard --logdir experiment_output/tb --bind_all --port
python3.6 main.py -o experiment_output -b projects/armnav_baselines/experiments/ithor/ armnav_ithor_rgb_simplegru_ddppo
#ssh -NfL 6015:localhost:6015 aws15;ssh -NfL 6016:localhost:6016 aws16;ssh -NfL 6014:localhost:6014 aws14;ssh -NfL 6017:localhost:6017 aws17;
