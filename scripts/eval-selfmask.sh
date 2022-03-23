#!/usr/bin/bash

gpu_id=0
config="/users/gyungin/selfmask/configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"
p_state_dict="/users/gyungin/selfmask_bak/ckpt/nq20_ndl6_bc_sr10100_duts_pm_all_k2,3,4_md_seed0_final/eval/duts/best_model.pt"
dataset_name="ecssd"

python3 ../evaluator.py --gpu_id "${gpu_id}" --config "${config}" --p_state_dict "${p_state_dict}" -dn "${dataset_name}"