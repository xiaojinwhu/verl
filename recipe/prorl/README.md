# ProRL: Prolonged Reinforcement Learning

Implementation of **ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models** in the verl framework.

## Overview

ProRL is a novel training methodology that demonstrates prolonged RL training can uncover novel reasoning strategies inaccessible to base models, even under extensive sampling. This implementation includes:

- **KL Divergence Control**: Dynamic KL penalty with adaptive coefficients
- **Reference Policy Resetting**: Periodic hard reset for prolonged training stability
- **DAPO-style Enhancements**: Decoupled clipping and dynamic sampling for improved exploration
- **Diverse Task Training**: Support for math, code, STEM, logic puzzles, and instruction following

## Key Features

### 🚀 Core ProRL Components

1. **Enhanced GRPO**: Group Relative Policy Optimization with ProRL modifications
2. **Dynamic KL Control**: Adaptive KL divergence penalties to prevent policy collapse
3. **Reference Policy Reset**: Periodic hard reset of reference policy and optimizer states
4. **Entropy Management**: Prevents entropy collapse during prolonged training

### 🎯 DAPO Integration

- **Decoupled Clipping**: Separate `clip_ratio_low` and `clip_ratio_high` for better exploration
- **Dynamic Sampling**: Filters groups with uniform rewards to maintain learning signal
- **Token-level Loss**: Enhanced loss aggregation for stable training
- **Overlong Reward Shaping**: Prevents artificially long responses

### 📊 Diverse Task Support

- **Math Reasoning**: GSM8K, MATH-style problems
- **Code Generation**: HumanEval, MBPP-style tasks
- **STEM Reasoning**: Physics, chemistry, biology problems
- **Logic Puzzles**: Pattern recognition, constraint satisfaction
- **Instruction Following**: Format, length, style constraints

## Quick Start

### 1. Environment Setup

```bash
# Ensure you have verl installed
cd /path/to/verl
pip install -e .

# Navigate to ProRL recipe
cd recipe/prorl
```

### 2. Data Preparation

```bash
# Create diverse task datasets
python prepare_prorl_data.py --output_dir data/prorl_diverse_tasks

# This creates:
# - math_problems.jsonl (10K samples)
# - code_problems.jsonl (5K samples)  
# - stem_problems.jsonl (3K samples)
# - logic_puzzles.jsonl (2K samples)
# - instruction_following.jsonl (3K samples)
# - validation_data.jsonl (1K samples)
```

### 3. Training

```bash
# Run ProRL training on Qwen 1.5B
bash run_prorl_qwen_1.5b.sh

# Or customize your training
python main_prorl.py \
    trainer.experiment_name="my_prorl_experiment" \
    data.train_batch_size=256 \
    algorithm.reference_reset.interval=100 \
    actor_rollout_ref.model.path="/path/to/your/model"
```

## Configuration

### Key Parameters

```yaml
# KL Divergence Control
algorithm:
  kl_control:
    initial_coef: 0.01      # Starting KL coefficient
    target_kl: 0.05         # Target KL divergence
    adaptive: true          # Enable adaptive adjustment
    min_coef: 0.001         # Minimum coefficient
    max_coef: 0.1           # Maximum coefficient

# Reference Policy Reset
  reference_reset:
    interval: 100           # Reset every N steps
    kl_threshold: 0.1       # Reset if KL exceeds threshold
    performance_degradation: true  # Reset on validation degradation

# DAPO Enhancements
  filter_groups:
    enable: true            # Dynamic sampling
    metric: "acc"           # Metric for filtering
    max_num_gen_batches: 10 # Max generation attempts

# Actor Configuration
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2     # Lower clip bound
    clip_ratio_high: 0.28   # Higher clip bound (clip-higher)
    use_kl_loss: true       # Direct KL loss
    loss_agg_mode: "token-mean"  # Token-level aggregation
```

### Prolonged Training Settings

```yaml
# Extended training configuration
trainer:
  total_epochs: 20          # Extended training period
  save_freq: 100           # Frequent checkpointing
  val_freq: 50             # Regular validation

# Context window expansion
data:
  max_response_length: 8192  # Start with 8K
  # Expand to 16K during training

# High exploration settings
actor_rollout_ref:
  rollout:
    n: 16                  # Group sampling size
    temperature: 1.2       # High temperature for exploration
```

## Architecture

### ProRL Trainer Flow

```
Input Prompts
     ↓
Group Sampling (n=16 per prompt)
     ↓
Dynamic Filtering (remove uniform groups)
     ↓
GRPO Advantage Estimation
     ↓
KL Divergence Control
     ↓
Policy Update (with decoupled clipping)
     ↓
Reference Reset Check
     ↓
Validation & Logging
```

### Key Classes

- `RayProRLTrainer`: Main trainer with ProRL enhancements
- `ReferencyPolicyResetController`: Manages reference policy resets
- `DynamicKLController`: Adaptive KL divergence control
- `ProRLEntropyController`: Prevents entropy collapse

## Results Monitoring

### ProRL-specific Metrics

```python
# Training metrics logged automatically
{
    'prorl/kl_divergence': 0.045,          # Current KL divergence
    'prorl/kl_coefficient': 0.012,         # Adaptive KL coefficient
    'prorl/reset_count': 3,                # Number of resets performed
    'prorl/steps_since_reset': 87,         # Steps since last reset
    'prorl/entropy': -2.1,                 # Current entropy
    'prorl/entropy_trend': -0.05,          # Entropy change trend
    'prorl/avg_response_length': 234.5,    # Average response length
}
```

### Validation Metrics

The trainer automatically validates on diverse tasks and tracks:
- Task-specific accuracy (math, code, STEM, logic, instruction)
- Pass@k metrics for reasoning tasks
- Response quality and length distributions
- Creativity index for novel solution generation

## Advanced Usage

### Custom Task Integration

```python
# Add your own task type to the reward manager
def custom_task_reward(response, ground_truth, task_metadata):
    # Your custom reward logic
    return reward_score

# Register in reward manager configuration
```

### Hyperparameter Tuning

Key hyperparameters for optimization:

1. **KL Control**: Adjust `initial_coef` and `target_kl` based on model behavior
2. **Reset Frequency**: Tune `reference_reset.interval` for stability vs. exploration
3. **Clipping Bounds**: Balance `clip_ratio_low/high` for learning stability
4. **Group Sampling**: Optimize `rollout.n` for your computational budget

### Multi-Node Training

```bash
# Update configuration for multi-node setup
trainer:
  nnodes: 4                # Number of nodes
  n_gpus_per_node: 8       # GPUs per node

# Launch with Ray cluster
export RAY_ADDRESS="ray://head-node:10001"
python main_prorl.py ...
```

## Troubleshooting

### Common Issues

1. **Entropy Collapse**: Increase `entropy_coeff` or reduce `kl_loss_coef`
2. **Training Instability**: Reduce `clip_ratio_high` or increase reset frequency
3. **Memory Issues**: Reduce `rollout.n` or `max_response_length`
4. **Slow Convergence**: Increase learning rate or adjust KL target

### Performance Tips

- Use `loss_agg_mode: "token-mean"` for stable training
- Enable `filter_groups` for better sample efficiency
- Monitor entropy trends and adjust controllers accordingly
- Use validation metrics to guide reset timing

## Citation

If you use this implementation, please cite:

```bibtex
@article{liu2025prorl,
  title={ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models},
  author={Liu, Mingjie and Diao, Shizhe and Lu, Ximing and others},
  journal={arXiv preprint arXiv:2505.24864},
  year={2025}
}
```

## Acknowledgments

This implementation is built on the [verl](https://github.com/volcengine/verl) framework and incorporates ideas from:
- **GRPO**: Group Relative Policy Optimization
- **DAPO**: Decoupled Clip and Dynamic Sampling Policy Optimization
- **DeepSeek-R1**: Reasoning model architecture and training

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.