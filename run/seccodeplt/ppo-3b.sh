#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

# PROJECT_NAME=training-safety-code-rl
PROJECT_NAME=test
EXPERIMENT_NAME=grpo_qwen2.5-3b-coder-instruct-hybrid-test

TRAIN_DATA=./data/seccodeplt/train.parquet
VAL_DATA=./data/seccodeplt/test.parquet
BASE_MODEL=/mnt/data/safetyCode/ckpts/SFT/qwen25coder3b_sft_safe_traj/checkpoint-564

# Auto-detect number of GPUs from CUDA_VISIBLE_DEVICES
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Reward Configuration - Choose one by uncommenting
# REWARD_CONFIG="detector_only"     # detector only
# REWARD_CONFIG="unit_test_only"    # unit test only
# REWARD_CONFIG="detector_safety"   # detector + safety unit test
REWARD_CONFIG="hybrid"              # hybrid mix (current: scpd=0.5, ut=0.5)

export WANDB_API_KEY="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"
export VLLM_ATTENTION_BACKEND=XFORMERS

#  critic.optim.lr=1e-5 \
#  critic.model.path=$BASE_MODEL \
#  critic.ppo_micro_batch_size_per_gpu=2 \
# ---------------------------------------------------------------------------

TIME_TAG=$(date +%Y%m%d-%H%M%S)
MOO_ALGORITHM="DWA"
# Set reward ratios based on configuration
case $REWARD_CONFIG in
    "detector_only")
        SAFETY_RATIO=0.0
        MYPY_RATIO=0.0
        SCPD_RATIO=1.0
        UT_RATIO=0.0
        CONFIG_DESC="Detector Only"
        ;;
    "unit_test_only")
        SAFETY_RATIO=0.0
        MYPY_RATIO=0.0
        SCPD_RATIO=0.0
        UT_RATIO=1.0
        CONFIG_DESC="Unit Test Only"
        ;;
    "detector_safety")
        SAFETY_RATIO=0.5
        MYPY_RATIO=0.0
        SCPD_RATIO=0.0
        UT_RATIO=1.0
        CONFIG_DESC="Detector + Safety Unit Test"
        ;;
    "hybrid")
        SAFETY_RATIO=0.0
        MYPY_RATIO=0.0
        SCPD_RATIO=0.5
        UT_RATIO=0.5
        CONFIG_DESC="Hybrid (SCPD + Unit Test)"
        ;;
    *)
        echo "Invalid REWARD_CONFIG: $REWARD_CONFIG"
        exit 1
        ;;
esac

echo "----------------------------------------"
echo "Project: $PROJECT_NAME"
echo "Experiment: $EXPERIMENT_NAME-$TIME_TAG"
echo "Base Model: $BASE_MODEL"
echo "Train Data: $TRAIN_DATA"
echo "Validation Data: $VAL_DATA"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $N_GPUS"
echo "----------------------------------------"
echo "Reward Configuration: $CONFIG_DESC"
echo "MOO Algorithm: $MOO_ALGORITHM"
echo "Safety Ratio: $SAFETY_RATIO"
echo "Mypy Ratio: $MYPY_RATIO"
echo "SCPD Ratio: $SCPD_RATIO"
echo "UT Ratio: $UT_RATIO"
echo "----------------------------------------"

read -p "Do you want to continue? (y/n): " choice
if [[ $choice != "y" ]]; then
    echo "Exiting..."
    exit 1
fi

mkdir -p ./log/$PROJECT_NAME

# ---------------------------------------------------------------------------

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=$TRAIN_DATA \
 data.val_files=$VAL_DATA \
 algorithm.adv_estimator=gae \
 data.train_batch_size=512 \
 data.val_batch_size=512 \
 data.max_prompt_length=375 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 critic.optim.lr=1e-5 \
 critic.model.path=$BASE_MODEL \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','wandb'] \
 +trainer.val_before_train=True \
 trainer.val_generations_to_log_to_wandb=5 \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.n_gpus_per_node=4 \
 trainer.save_freq=20 \
 trainer.test_freq=1 \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME-$TIME_TAG \
 trainer.resume_from_path=True \
 trainer.total_epochs=140 \
 +safety_ratio=$SAFETY_RATIO \
 trainer.default_local_dir=/mnt/data/safetyCode/ckpts/$PROJECT_NAME/$EXPERIMENT_NAME-$TIME_TAG \
 +mypy_ratio=$MYPY_RATIO \
 +scpd_ratio=$SCPD_RATIO \
 +ut_ratio=$UT_RATIO \
 +moo_algorithm=$MOO_ALGORITHM \
 2>&1 | tee ./log/$PROJECT_NAME/$EXPERIMENT_NAME-$TIME_TAG.log