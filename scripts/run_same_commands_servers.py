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

# command_aws1 = 'scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORMultipleRoomsDistrib/procthor_armpointnav_rgbonly_multiple_rooms/2022-05-13_17-23-45/exp_ObjDisArmPointNavRGBOnlyProcTHORMultipleRoomsDistrib_procthor_armpointnav_rgbonly_multiple_rooms__stage_00__steps_000011242397.pt ~/;  \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_rgb_only_multiple_rooms_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_armpointnav_rgbonly_multiple_rooms -c ~/exp_ObjDisArmPointNavRGBOnlyProcTHORMultipleRoomsDistrib_procthor_armpointnav_rgbonly_multiple_rooms__stage_00__steps_000011242397.pt'


# command_aws1 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib/clip_obj_dis_for_procthor_rgb_only_distrib/2022-05-14_15-09-10/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_clip_obj_dis_for_procthor_rgb_only_distrib__stage_00__steps_000076285861.pt ~/;  \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/clip_obj_dis_for_procthor_rgb_only_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag clip_obj_dis_for_procthor_rgb_only_distrib -c ~/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_clip_obj_dis_for_procthor_rgb_only_distrib__stage_00__steps_000076285861.pt'

# command = command_aws1
# scp 18.237.24.199:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORDistrib/procthor_armpointnav_rgbonly/2022-05-11_17-16-05/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_procthor_armpointnav_rgbonly__stage_00__steps_000058633471.pt ~/

#
# command_aws5 = 'scp 52.24.154.159:~/manipulathor/experiment_output/checkpoints/CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib/armpointnav_with_clip_ProcTHOR_RGBonly/2022-05-12_04-54-18/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_armpointnav_with_clip_ProcTHOR_RGBonly__stage_00__steps_000005061351.pt ~/; \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/clip_obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag armpointnav_with_clip_ProcTHOR_RGBonly -c ~/exp_CLIPObjDisArmPointNavProcTHORAllRoomsRGBOnlyDistrib_armpointnav_with_clip_ProcTHOR_RGBonly__stage_00__steps_000005061351.pt'

# command_aws1 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavITHORAllRoomsRGBOnlyDistrib/armpointnav_with_iTHOR_RGBonly_after_fix/2022-05-16_21-15-02/exp_ObjDisArmPointNavITHORAllRoomsRGBOnlyDistrib_armpointnav_with_iTHOR_RGBonly_after_fix__stage_00__steps_000061755450.pt ~/&&\
# ./manipulathor/scripts/kill-zombie.sh && cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_ithor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag armpointnav_with_iTHOR_RGBonly_after_fix -c ~/exp_ObjDisArmPointNavITHORAllRoomsRGBOnlyDistrib_armpointnav_with_iTHOR_RGBonly_after_fix__stage_00__steps_000061755450.pt'

# command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavRGBOnlyProcTHORDistrib/armpointnav_with_ProcTHOR_RGBonly_after_fix/2022-05-14_00-37-27/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_armpointnav_with_ProcTHOR_RGBonly_after_fix__stage_00__steps_000018363080.pt ~/; \
# ./manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_rgb_only_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag armpointnav_with_ProcTHOR_RGBonly_after_fix -c ~/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_armpointnav_with_ProcTHOR_RGBonly_after_fix__stage_00__steps_000018363080.pt'
#
# command_aws1 = './manipulathor/scripts/kill-zombie.sh ; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/pnemul_obj_dis_for_ithor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag objdis_pointnav_emul_ithor'
# command_aws5 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/PNEmulObjDisArmPointNavProcTHORAllRoomsDistrib/objdis_pointnav_emul_procthor/2022-05-17_20-37-49/exp_PNEmulObjDisArmPointNavProcTHORAllRoomsDistrib_objdis_pointnav_emul_procthor__stage_00__steps_000004028790.pt ~/; ./manipulathor/scripts/kill-zombie.sh ; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/pnemul_obj_dis_for_procthor_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag objdis_pointnav_emul_procthor -c ~/exp_PNEmulObjDisArmPointNavProcTHORAllRoomsDistrib_objdis_pointnav_emul_procthor__stage_00__steps_000004028790.pt'

# command_aws8 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavProcTHORDistrib/procthor_rgbd_apointnav_restart/2022-08-02_15-04-11/exp_ObjDisArmPointNavProcTHORDistrib_procthor_rgbd_apointnav_restart__stage_00__steps_000045807530.pt ~/; ./manipulathor/scripts/kill-zombie.sh ; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag procthor_rgbd_apointnav_restart --enable_crash_recovery -c ~/exp_ObjDisArmPointNavProcTHORDistrib_procthor_rgbd_apointnav_restart__stage_00__steps_000045807530.pt'


command_aws8 = 'scp MAIN_SERVER:~/manipulathor/experiment_output/checkpoints/ObjDisArmPointNavProcTHORDistrib/procthor_only_depth_apointnav/2022-08-05_23-50-23/exp_ObjDisArmPointNavProcTHORDistrib_procthor_only_depth_apointnav__stage_00__steps_000007036090.pt ~/ ;./manipulathor/scripts/kill-zombie.sh ; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_procthor_distrib.py \
--distributed_ip_and_port IP_ADR:6060 \
 --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
 --seed 10 --machine_id 0 --extra_tag procthor_only_depth_apointnav --enable_crash_recovery -c ~/exp_ObjDisArmPointNavProcTHORDistrib_procthor_only_depth_apointnav__stage_00__steps_000007036090.pt'
#
#
# command_aws5 = './manipulathor/scripts/kill-zombie.sh ; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/procthor_baselines/experiments/ithor/obj_dis_for_ithor_rgb_only_distrib.py \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag finetune_procthor_on_ithor_smaller_lr_4times_b500 -c ~/exp_ObjDisArmPointNavRGBOnlyProcTHORDistrib_armpointnav_with_ProcTHOR_RGBonly_after_fix__stage_00__steps_000098567812.pt'

command_aws1 = ''
command_aws5 = ''
command_aws15 = ''
command_vs411 = ''
command = None

# command = command_aws5
# command = './manipulathor/scripts/kill-zombie.sh; cd manipulathor && export PYTHONPATH="./" && allenact manipulathor_baselines/stretch_bring_object_baselines/experiments/ithor/no_pointnav_stretch_all_rooms_distrib \
# --distributed_ip_and_port IP_ADR:6060 \
#  --config_kwargs \'{\\"distributed_nodes\\":NUM_MACHINES}\' \
#  --seed 10 --machine_id 0 --extra_tag ithor_train_no_armpointnav -c ~/exp_NoPointNavStretchAllRoomsDistrib_ithor_train_no_armpointnav__stage_00__steps_000114436451.pt'

# scp 52.24.154.159:~/manipulathor/experiment_output/checkpoints/NoPointNavStretchAllRoomsDistrib/ithor_train_no_armpointnav/2022-05-09_22-13-53/exp_NoPointNavStretchAllRoomsDistrib_ithor_train_no_armpointnav__stage_00__steps_000114436451.pt ~/
server_mapping = dict(
    aws1 = {
        'servers':[f'aws{i}' for i in range(1,5)],
        'ip_adr': '18.237.24.199',
        'command': command_aws1,
    },
    aws5 = {
        'servers':[f'aws{i}' for i in range(5, 8)],
        'ip_adr': '52.24.154.159',
        'command': command_aws5,
    },
    aws8 = {
        'servers':[f'aws{i}' for i in range(8, 12)],
        'ip_adr': '35.90.135.47',
        'command': command_aws8,
    },
    aws15 = {
        'servers':[f'aws{i}' for i in range(1, 9)],
        'ip_adr': '18.237.24.199',
        'command': command_aws15,
    },

    vs411 = {
        'servers':['vision-server11', 'vision-server4'],
        'ip_adr': '172.16.122.186',
        'command': command_vs411,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--server_set', default=None)#, nargs='+')
    parser.add_argument('--command', default=command, type=str)
    parser.add_argument('--directly', action='store_true')

    args = parser.parse_args()

    args.servers = []
    info_for_server = server_mapping[args.server_set]
    args.servers = info_for_server['servers']
    ip_adr = info_for_server['ip_adr']
    if args.command is None:
        args.command = info_for_server['command']

    args.command = args.command.replace('MAIN_SERVER', ip_adr)
    args.command = args.command.replace('IP_ADR', ip_adr)
    args.command = args.command.replace('NUM_MACHINES', str(len(args.servers)))
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


