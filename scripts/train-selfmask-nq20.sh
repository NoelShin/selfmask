#!/usr/bin/bash

gpu_id=2
config="/users/gyungin/selfmask/configs/duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml"

python3 ../main.py --config "$config" --suffix '' --gpu_id "${gpu_id}"