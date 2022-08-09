import argparse
import os
import pdb

# command = './manipulathor/scripts/kill-zombie.sh'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_randomization_distrib \
#   --distributed_ip_and_port 34.220.30.46:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_memory_distrib \
#   --distributed_ip_and_port 34.220.30.46:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 -c ~/exp_ComplexRewardNoPUWMemory__stage_00__steps_000045112992.pt'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_binary_distance_distrib \
#   -c ~/exp_ComplexRewardNoPUBinaryDistanceDistrib__stage_00__steps_000007082384.pt \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_pointnav_distrib \
#   --distributed_ip_and_port IP_ADR:6060 --extra_tag with_abs_distance\
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/temp_test_pointnav_complex_reward_no_pu_pointnav_distrib \
#   --distributed_ip_and_port IP_ADR:6060 --extra_tag test_new_model_and_hand\
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/temp_test_train_real_pointnav_complex_reward_no_pu_pointnav_distrib \
#   --distributed_ip_and_port IP_ADR:6060 --extra_tag test_same_model_real_pointnav\
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_binary_distance_w_noise_and_discrim_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_distrib \
#   --extra_tag complex_reward_no_pu_w_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_w_pu_w_distrib \
#   --extra_tag complex_reward_w_pu_w_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_pc_w_distrib \
#   --extra_tag complex_reward_no_pu_w_pc_w_distrib \
#   -c ~/exp_ComplexRewardNoPUWPointCloudMemoryDistrib_complex_reward_no_pu_w_pc_w_distrib__stage_00__steps_000022787017.pt \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/predict_mask_rgbd_distrib \
#   --extra_tag predict_mask_rgbd_distrib \
#   -c ~/exp_PreditMaskDistrib_predict_mask_rgbd_distrib__stage_00__steps_000005013430.pt \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_memory_noise_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0 -c ~/exp_ComplexRewardNoPUWMemoryNoiseDistrib__stage_00__steps_000034923191.pt'
#
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0'

#
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_complex_reward_no_pu_w_binary_dist_distrib \
#   --distributed_ip_and_port IP_ADR:6060 \
#   --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#   --seed 10 --machine_id 0'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  -c ~/exp_ComplexRewardNoPUDistrib__stage_00__steps_000031802009.pt\
#  --seed 10 --machine_id 0'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_complex_reward_no_pu_w_noise_distrib \
#  --distributed_ip_and_port IP_ADR:6060 -c ~/exp_PointNavNewModelAndHandWAgentNoiseDistrib__stage_00__steps_000014534520.pt \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  --seed 10 --machine_id 0'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_no_mask_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  --seed 10 --machine_id 0'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_real_pointnav_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  --seed 10 --machine_id 0'


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/complex_reward_no_pu_w_agent_location_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  --seed 10 --machine_id 0'



# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_complex_reward_no_pu_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":4}\' \
#  --seed 10 --machine_id 0'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_only_agent_loc_complex_reward_no_pu_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_complex_reward_no_pu_distrib \
#  --distributed_ip_and_port IP_ADR:6060 -c ~/exp_TmpComplexRewardPointNavNoPUNewModelAndHandDistrib_test_new_model_and_hand__stage_00__steps_000046743650.pt\
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/pointnav_complex_reward_no_pu_zero_shot_distrib \
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/zero_shot_complex_reward_no_pu_w_agent_location_distrib -c ~/exp_ComplexRewardNoPUwAgentLocationZeroShotDistrib__stage_00__steps_000018553490.pt\
#  --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/real_point_nav_stretch_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag continue_from_pickup -c ~/exp_RealPointNavStretchDistrib__stage_00__steps_000025933161.pt'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_pickup_with_mask'# -c ~/exp_PointNavEmulStretchDistrib_train_object_nav_with_mask__stage_00__steps_000002540050.pt'

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_full_pretrain_pickup_with_mask -c ~/pickup_weight_000015685630.pt'



# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_only_pickup_all_ithor_rooms_scratch '


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_full_task_all_ithor_rooms_from_pickup -c ~/exp_PointNavEmulStretchAllRoomsDistrib_train_only_pickup_all_ithor_rooms_scratch__stage_00__steps_000016639355.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_full_task_distant_masks_only_from_pretrained -c ~/exp_PointNavEmulStretchAllRoomsDistrib_train_full_task_all_ithor_rooms_from_pickup__stage_00__steps_000100608660.pt '


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_full_task_distant_masks_only_pickup_with_realisticcamera_from_pretrained -c ~/exp_PointNavEmulStretchAllRoomsDistrib_train_full_task_all_ithor_rooms_from_pickup__stage_00__steps_000100608660.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag train_full_task_distant_masks_with_realisticcamera_from_pretrained -c ~/exp_PointNavEmulStretchAllRoomsDistrib_train_full_task_distant_masks_only_pickup_with_realisticcamera_from_pretrained__stage_00__steps_000165883510.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_robothor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag only_pickup_robothor_fixed_depth_ranges_from_pretrain -c ~/exp_PointNavEmulStretchAllRoomsDistrib_train_full_task_distant_masks_only_pickup_with_realisticcamera_from_pretrained__stage_00__steps_000165883510.pt '


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_robothor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag only_pickup_robothor_fixed_depth_ranges_nomindepth_from_pretrain -c ~/exp_PointNavEmulStretchRoboTHORDistrib_only_pickup_robothor_fixed_depth_ranges_from_pretrain__stage_00__steps_000289507150.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_robothor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag only_pickup_robothor_wide_range_depth_from_pretrain -c ~/exp_PointNavEmulStretchRoboTHORDistrib_only_pickup_robothor_fixed_depth_ranges_from_pretrain__stage_00__steps_000289507150.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_only_pickup_from_pretrain -c ~/exp_PointNavEmulStretchRoboTHORDistrib_only_pickup_robothor_fixed_depth_ranges_from_pretrain__stage_00__steps_000289507150.pt '
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_only_pickup_from_pretrain -c ~/exp_PointNavEmulStretchProcTHORDistrib_procthor_onlypickup_task_scratch__stage_00__steps_000010595298.pt '

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/no_pointnav_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag ithor_train_no_armpointnav'
#  --seed 10 --machine_id 0 --extra_tag procthor_only_pickup_from_pretrain_big_fov -c ~/exp_PointNavEmulStretchRoboTHORDistrib_only_pickup_robothor_fixed_depth_ranges_from_pretrain__stage_00__steps_000289507150.pt '

#


# scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/PointNavEmulStretchProcTHORDistrib/procthor_onlypickup_task_scratch/2022-04-28_16-56-04/exp_PointNavEmulStretchProcTHORDistrib_procthor_onlypickup_task_scratch__stage_00__steps_000010595298.pt ~/
# pip3 install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5afa5633597b12898e12eed528c2332a50bc0f79

# scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/PointNavEmulStretchAllRoomsDistrib/train_only_pickup_all_ithor_rooms_scratch/2022-03-03_21-40-29/exp_PointNavEmulStretchAllRoomsDistrib_train_only_pickup_all_ithor_rooms_scratch__stage_00__steps_000016639355.pt ~/

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/real_point_nav_stretch_objectnav_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'

# scp 18.237.159.252:~/manipulathor/experiment_output/checkpoints/ComplexRewardNoPUwAgentLocationZeroShotDistrib/2021-11-12_19-42-56/exp_ComplexRewardNoPUwAgentLocationZeroShotDistrib__stage_00__steps_000018553490.pt ~/
# command = 'scp ec2-34-220-30-46.us-west-2.compute.amazonaws.com:~/manipulathor/experiment_output/checkpoints/ComplexRewardNoPUWMemory/2021-10-08_23-12-59/exp_ComplexRewardNoPUWMemory__stage_00__steps_000045112992.pt ~/'
# list_of_servers = ['aws1', 'aws2', 'aws3', 'aws4', ]

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_onlypickup_task_scratch -c ~/exp_PointNavEmulStretchProcTHORDistrib_procthor_onlypickup_task_scratch__stage_00__steps_000019677434.pt'

# scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/PointNavEmulStretchProcTHORDistrib/procthor_onlypickup_task_scratch/2022-04-29_16-27-40/exp_PointNavEmulStretchProcTHORDistrib_procthor_onlypickup_task_scratch__stage_00__steps_000019677434.pt ~/

# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/pointnav_emul_stretch_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_onlypickup_task_scratch_after_debug'


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_armpointnav -c ~/exp_ObjDisArmPointNavProcTHORDistrib_procthor_armpointnav__stage_00__steps_000001006560.pt'


# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_armpointnav_rgbonly'

# command_aws1 = 'scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORDistrib/procthor_armpointnav_rgbonly/2022-05-11_20-52-20/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_procthor_armpointnav_rgbonly__stage_00__steps_000106644058.pt ~/;  \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_armpointnav_rgbonly -c ~/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_procthor_armpointnav_rgbonly__stage_00__steps_000106644058.pt'

# command_aws1 = 'scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORDistrib/procthor_armpointnav_rgbonly/2022-05-11_20-52-20/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_procthor_armpointnav_rgbonly__stage_00__steps_000106644058.pt ~/;  \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/clip_obj_dis_for_procthor_rgb_only_multiple_rooms_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_armpointnav_rgbonly_multiple_rooms_clip'

# command = command_aws1
# scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORDistrib/procthor_armpointnav_rgbonly/2022-05-11_17-16-05/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_procthor_armpointnav_rgbonly__stage_00__steps_000058633471.pt ~/

#
# command_aws5 = 'scp 52.24.154.159:~/manipulathor/experiment_output/checkpoints/CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib/armpointnav_with_clip_ProcTHOR_RGBonly/2022-05-12_04-54-18/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_armpointnav_with_clip_ProcTHOR_RGBonly__stage_00__steps_000005061351.pt ~/; \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/clip_obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag armpointnav_with_clip_ProcTHOR_RGBonly -c ~/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_armpointnav_with_clip_ProcTHOR_RGBonly__stage_00__steps_000005061351.pt'


# command_aws5 = 'scp 52.24.154.159:~/manipulathor/experiment_output/checkpoints/CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib/armpointnav_with_clip_ProcTHOR_RGBonly/2022-05-12_04-54-18/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_armpointnav_with_clip_ProcTHOR_RGBonly__stage_00__steps_000005061351.pt ~/; \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/clip_obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag armpointnav_with_ProcTHOR_RGBonly_after_fix_clip '

# command_aws5 = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_nav_2camera_procthor_wide_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag prcthor_obj_nav '

# command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/RobothorObjectNavClipResnet50RGBOnly2CameraNarrowFOVDistrib/finetune_robothor/2022-05-27_21-00-03/exp_RobothorObjectNavClipResnet50RGBOnly2CameraNarrowFOVDistrib_finetune_robothor__stage_02__steps_000160201650.pt ~/; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch/experiments/ithor/kiana_obj_nav_2camera_procthor_wide_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag kiana_objnav_outside_room_wide -c ~/exp_ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVDistrib_prcthor_obj_nav__stage_02__steps_000100123710.pt '

command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/StretchObjectNavTaskIntelSegmentationSuccess-RGB-1Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test/2022-07-26_22-15-49/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-1Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000062287440.pt   ~/; \
     ./manipulathor/scripts/kill-zombie.sh; sleep 5s; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_1camera_distrib.py \
--distributed_ip_and_port IP_ADR:6060 \
 --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
 --seed 10 --machine_id 0 -c ~/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-1Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000062287440.pt '

# command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/StretchNeckedObjectNavTaskUpdateOrder-RGB-SingleCam-ProcTHOR-locobot-RoboTHOR-Test/2022-06-08_21-59-41/exp_StretchNeckedObjectNavTaskUpdateOrder-RGB-SingleCam-ProcTHOR-locobot-RoboTHOR-Test__stage_01__steps_000014017504.pt  ~/; ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 -c ~/exp_StretchNeckedObjectNavTaskUpdateOrder-RGB-SingleCam-ProcTHOR-locobot-RoboTHOR-Test__stage_01__steps_000014017504.pt '

command_aws1 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test/2022-08-08_18-14-14/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000275660336.pt    ~/; \
     ./manipulathor/scripts/kill-zombie.sh; sleep 5s; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_2camera_distrib.py \
--distributed_ip_and_port IP_ADR:6060 \
 --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
 --seed 10 --machine_id 0 -c ~/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000275660336.pt '

# command_aws1 = ' ./manipulathor/scripts/kill-zombie.sh; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_2camera_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 '

# command_aws1 = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0'

# command = command_aws5
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/no_pointnav_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag ithor_train_no_armpointnav -c ~/exp_NoPointNavStretchAllRoomsDistrib_ithor_train_no_armpointnav__stage_00__steps_000114436451.pt'

# scp 52.24.154.159:~/manipulathor/experiment_output/checkpoints/NoPointNavStretchAllRoomsDistrib/ithor_train_no_armpointnav/2022-05-09_22-13-53/exp_NoPointNavStretchAllRoomsDistrib_ithor_train_no_armpointnav__stage_00__steps_000114436451.pt ~/


# command_awsv1 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test/2022-07-27_16-21-39/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000070733520.pt   ~/; \
#      ./manipulathor/scripts/kill-zombie.sh; sleep 5s; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_2camera_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 -c ~/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_02__steps_000070733520.pt '

command_awsv1 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test/2022-08-08_16-17-58/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_01__steps_000020004756.pt  ~/; \
     ./manipulathor/scripts/kill-zombie.sh; sleep 5s; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_2camera_distrib.py \
--distributed_ip_and_port IP_ADR:6060 \
 --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
 --seed 10 --machine_id 0 -c ~/exp_StretchObjectNavTaskIntelSegmentationSuccess-RGB-2Camera-ProcTHOR-narrowFOV-stretch-RoboTHOR-Test__stage_01__steps_000020004756.pt \
--enable_crash_recovery'

# command_awsv1 = './manipulathor/scripts/kill-zombie.sh; sleep 5s; ai2thor-xorg start; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_object_nav_baselines/experiments/robothor/test_robothor_procthorstyle_2camera_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 \
# --enable_crash_recovery'

server_sets = {
    'aws1':{
        'servers':[f'aws{i}' for i in range(1,5)],
        'ip_adr': '52.13.80.54',
        'command': command_aws1,
    },
    'aws5':  {
        'servers':[f'aws{i}' for i in range(5, 8)],
        'ip_adr': '34.216.219.227',
        'command': command_aws5,
    },
    'awsv1':{
        'servers':[f'awsv{i}' for i in range(1,5)],
        'ip_adr': '107.22.133.78',
        'command': command_awsv1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--server_set', default=None, nargs='+')
    parser.add_argument('--command', default=None, type=str)
    parser.add_argument('--directly', action='store_true')

    args = parser.parse_args()
    args.servers = []
    server = server_sets[args.server_set[0]]
    args.servers += server['servers']
    ip_adr = server['ip_adr']
    args.command  = server['command']
    args.command = args.command.replace('IP_ADR', ip_adr)
    args.command = args.command.replace('NUM_MACHINES', str(len(args.servers)))
    args.command = args.command.replace('MAIN_SERVER', ip_adr) 
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


