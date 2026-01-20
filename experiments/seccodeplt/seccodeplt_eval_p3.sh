# =========================== ReaL ============================
# You will first have to use `scripts/merge.sh` to merge your checkpoint
export CUDA_VISIBLE_DEVICES=3
full_path="/mnt/data/safetyCode/experiments/saved_checkpoints_compare/training-safety-code-rl/base_rl-20260118-182055/global_step_40/actor/huggingface"
python eval.py --model_name "$full_path" --cuda_idx 0 --batch_size 64