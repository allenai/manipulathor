#deep learning, g4dn.8xlarge or g4dn.4xlarge, change storage, change name, set public key, add to ssh config
cd manipulathor
echo "set-option -g history-limit 3000000" > ~/.tmux.conf
tmux
pip3 install --upgrade pip
pip3 install -r aws_requirements.txt
ssh-keygen; cat ~/.ssh/id_rsa.pub
cat scripts/public_keys.txt >> ~/.ssh/authorized_keys
#cat ~/manipulathor/scripts/public_keys.txt >> ~/.ssh/authorized_keys
sudo apt-get install xinit
sudo python3 scripts/startx.py &
tensorboard --logdir experiment_output/tb --bind_all
#sudo /home/ubuntu/.local/bin/ai2thor-xorg start
export PYTHONPATH="./"
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu
kill -9 $(ps aux | grep 'ssh -NfL'); ssh -NfL 6006:localhost:6006 aws1; #ssh -NfL 6007:localhost:6006 aws5; ssh -NfL 6008:localhost:6006 aws9;

#cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
#  --distributed_ip_and_port 34.220.30.46:6060 \
#  --config_kwargs '{"distributed_nodes":4}' \
#  --seed 10 --machine_id 0