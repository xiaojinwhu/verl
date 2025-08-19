#!/bin/bash
# ProRL Training Script for Qwen 1.5B Model
# Based on the paper: "ProRL: Prolonged Reinforcement Learning Expands 
# Reasoning Boundaries in Large Language Models"

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN

# Model and data paths
MODEL_PATH="models/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_DIR="data/prorl_diverse_tasks"
OUTPUT_DIR="outputs/prorl_qwen_1.5b"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting ProRL training for Qwen 1.5B..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# Run ProRL training
python main_prorl.py \
    trainer.project_name="ProRL_Qwen_1.5B" \
    trainer.experiment_name="prorl_diverse_reasoning_$(date +%Y%m%d_%H%M%S)" \
    trainer.total_epochs=20 \
    trainer.save_freq=100 \
    trainer.val_freq=50 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    \
    data.train_files=[\
"$DATA_DIR/math_problems.jsonl",\
"$DATA_DIR/code_problems.jsonl",\
"$DATA_DIR/stem_problems.jsonl",\
"$DATA_DIR/logic_puzzles.jsonl",\
"$DATA_DIR/instruction_following.jsonl"\
] \
    data.val_files=["$DATA_DIR/validation_data.jsonl"] \
    data.train_batch_size=256 \
    data.max_response_length=8192 \
    \
    algorithm.adv_estimator=prorl_grpo \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_control.initial_coef=0.01 \
    algorithm.kl_control.target_kl=0.05 \
    algorithm.kl_control.adaptive=true \
    algorithm.reference_reset.interval=100 \
    algorithm.reference_reset.kl_threshold=0.1 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric="acc" \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    \
    critic.model.path="$MODEL_PATH" \
    \
    reward_model.enable=false \
    reward_model.overlong_buffer.enable=true \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    \
    hydra.run.dir="$OUTPUT_DIR" \
    hydra.job.chdir=true

echo "ProRL training completed!"
echo "Results saved to: $OUTPUT_DIR"