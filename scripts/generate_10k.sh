#!/bin/bash
export PYTHONPATH="/home/jl_fs/workspace/pragmatic_calibration2:$PYTHONPATH"
python trained_calibration/rl/dataset/dpo_dataset.py --cfg trained_calibration/configs/dpo/mistral_create_10k.yaml
