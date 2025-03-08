#!/bin/bash
export PYTHONPATH="/home/tri/zhiyuan/yibo/pragmatic_calibration:$PYTHONPATH"
python trained_calibration/rl/dataset/dpo_dataset.py --cfg trained_calibration/configs/dpo/mistral_create_10k.yaml
