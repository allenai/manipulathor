#deep learning, g4dn.8xlarge or g4dn.4xlarge, change storage, change name, set public key, add to ssh config
cd manipulathor
tmux
pip3 install -r aws_requirements.txt
ssh-keygen; cat ~/.ssh/id_rsa.pub
sudo apt-get install xinit
sudo python3 scripts/startx.py &
cat scripts/public_keys.txt >> ~/.ssh/authorized_keys
export PYTHONPATH="./"
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu
tensorboard --logdir experiment_output/tb --bind_all
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
  --config_kwargs '{"distributed_nodes":4}' \
  --distributed_ip_and_port 34.220.30.46:6060 \
  --seed 10 --machine_id 0
ssh -NfL 6006:localhost:6006 aws1


#cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
#  --distributed_ip_and_port 34.220.30.46:6060 \
#  --config_kwargs '{"distributed_nodes":4}' \
#  --seed 10 --machine_id 0