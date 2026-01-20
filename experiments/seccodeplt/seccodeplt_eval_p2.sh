# =========================== ReaL ============================
# You will first have to use `scripts/merge.sh` to merge your checkpoint
export CUDA_VISIBLE_DEVICES=2
full_path="/mnt/data/safetyCode/experiments/saved_checkpoints_compare/training-safety-code-rl/base_rl-20260118-182055/global_step_20/actor/huggingface"
python eval.py --model_name "$full_path" --cuda_idx 0 --batch_size 64