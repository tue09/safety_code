# =========================== ReaL ============================
# You will first have to use `scripts/merge.sh` to merge your checkpoint
export CUDA_VISIBLE_DEVICES=1
full_path="/mnt/data/safetyCode/ckpts/training-safety-code-rl/grpo_qwen2.5-3b-coder-instruct-hybrid-balance-adv-famo-20260119-190509/global_step_80/actor/huggingface"
python eval.py --model_name "$full_path" --cuda_idx 0 --batch_size 64