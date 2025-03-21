
#!/bin/bash
export PYTHONPATH="$PWD:$PYTHONPATH"
llama_ckpt="../meta-llama/Llama-3.1-8B-Instruct"
python trained_calibration/rl/evaluate_dpo.py \
    --model ${llama_ckpt} \
    --trained_model $1 \
    --data_path "data/trivia_qa/tqa_validation_mistral_v1_small.jsonl" \
    --limit -1 \
    --out_path $(dirname ${1})/eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed 12 \
    --threshold 0.66
