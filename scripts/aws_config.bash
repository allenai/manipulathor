#deep learning, g4dn.8xlarge or g4dn.4xlarge, change storage, change name, set public key, add to ssh config
cd manipulathor
echo "set-option -g history-limit 3000000" > ~/.tmux.conf
tmux
python3 -m venv manipulathor_env
source ~/manipulathor_env/bin/activate
pip3 install --upgrade pip
pip3 install -r aws_requirements.txt
#ai2thor==0+dc0f9ecd8672dc2d62651f567ff95c63f3542332
#pip3 install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5afa5633597b12898e12eed528c2332a50bc0f79
pip3 install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+dc0f9ecd8672dc2d62651f567ff95c63f3542332
#ai2thor==0+dc0f9ecd8672dc2d62651f567ff95c63f3542332
#pip3 install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+08ec97f5d93486460f388b50d4ef5485468dc87f
ssh-keygen; cat ~/.ssh/id_rsa.pub
cat ~/manipulathor/scripts/public_keys.txt >> ~/.ssh/authorized_keys
#cat ~/manipulathor/scripts/public_keys.txt >> ~/.ssh/authorized_keys
sudo apt-get install xinit
sudo python3 scripts/startx.py &
tensorboard --logdir experiment_output/tb --bind_all
#sudo /home/ubuntu/.local/bin/ai2thor-xorg start
#sudo /home/kianae/manipulathor_env/bin/ai2thor-xorg start
export PYTHONPATH="./"
./scripts/kill-zombie.sh
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu
kill -9 $(ps aux | grep 'ssh -NfL'); ssh -NfL 6006:localhost:6006 aws1; #ssh -NfL 6007:localhost:6006 aws5; ssh -NfL 6008:localhost:6006 aws9;

pip3 uninstall clip; pip3 install git+https://github.com/openai/CLIP.git
pip3 uninstall allenact allenact_plugins; pip3 install -e "git+https://github.com/allenai/allenact.git@timeout-restart-true#egg=allenact&subdirectory=allenact"
# remove previous hugging dataset for procthor
#rm -rf ~/.cache/huggingface

#For installing dataset
#pip3 install datasets
#huggingface-cli login
#rm -rf ~/.cache/huggingface

#For solving the segmentation lazy backward uncompatibiliity
pip3 install --extra-index-url https://ai2thor-pypi.allenai.org  ai2thor==0+dc0f9ecd8672dc2d62651f567ff95c63f3542332

#cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
#  --distributed_ip_and_port 34.220.30.46:6060 \
#  --config_kwargs '{"distributed_nodes":4}' \
#  --seed 10 --machine_id 0


sudo service docker stop
sudo service gdm stop
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*


sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
#blacklist nouveau
#options nouveau modeset=0
sudo update-initramfs -u
sudo reboot
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run

pip3 install -e "git+https://github.com/allenai/allenact.git@307d3eaef1198456778a15268864c23a8b31b843#egg=allenact&subdirectory=allenact"
pip3 install -e "git+https://github.com/allenai/allenact.git@307d3eaef1198456778a15268864c23a8b31b843#egg=allenact_plugins[ithor]&subdirectory=allenact_plugins"
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
sudo /media/4TBNVME/home/kianae/some_virtual_env/bin/ai2thor-xorg start
sudo /media/4TBNVME/home/kianae/manipulathor_env/bin/ai2thor-xorg start
unset CUDA_VISIBLE_DEVICES
allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch

pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5afa5633597b12898e12eed528c2332a50bc0f79
#pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5f88945385c2bdcd1fe0a36069e92a48632dbc5b