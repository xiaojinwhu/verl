# GSPO with Dynamic Sampling

This directory contains the implementation of GSPO (Group Self-Play Optimization) algorithm integrated with DAPO's dynamic sampling mechanism. This combination improves data efficiency by dynamically filtering high-quality training examples based on reward variance.

## Overview

### GSPO Algorithm
GSPO is a policy optimization algorithm that uses dual clipping ratios and sequence-level importance sampling. Key features:
- **Dual Clipping**: Uses both low and high clipping ratios (`clip_ratio_low`, `clip_ratio_high`)
- **Sequence-level Importance Ratio**: Computes importance ratios at the sequence level rather than token level
- **Better Aggregation**: Recommended to use `seq-mean-token-mean` for loss aggregation

### Dynamic Sampling
Dynamic sampling, originally from DAPO, filters training samples based on reward quality:
- **Variance-based Filtering**: Keeps prompts with reward variance > 0 across multiple responses
- **Batch Size Adaptation**: Automatically generates additional batches if filtered samples < target batch size  
- **Quality Metrics**: Supports multiple filtering metrics (`seq_reward`, `seq_final_reward`, `acc`)

## Quick Start

### 1. Basic Training Script
```bash
bash recipe/gspo/gspo_dynamic_sampling_3b_math.sh
```

### 2. Python Training Script
```bash
cd recipe/gspo
python main_gspo_dynamic.py
```

### 3. Custom Configuration
```bash
python main_gspo_dynamic.py \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=seq_reward \
    algorithm.filter_groups.max_num_gen_batches=5 \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004
```

## Configuration Parameters

### GSPO-specific Parameters

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo  # Enable GSPO loss function
    clip_ratio_low: 0.0003    # Lower clipping ratio (recommended: 0.0003)
    clip_ratio_high: 0.0004   # Higher clipping ratio (recommended: 0.0004) 
    loss_agg_mode: "seq-mean-token-mean"  # Sequence-level aggregation
```

### Dynamic Sampling Parameters

```yaml
algorithm:
  filter_groups:
    enable: true                    # Enable dynamic sampling
    metric: "seq_reward"           # Filtering metric: seq_reward, seq_final_reward, acc
    max_num_gen_batches: 5         # Maximum generation attempts per training batch
```

### Recommended Settings

**For Math/Code Tasks:**
- `filter_groups.metric: "seq_reward"` - Use raw reward scores
- `max_num_gen_batches: 3-5` - Balance efficiency and quality
- `n_resp_per_prompt: 16` - Generate multiple responses for filtering

**For General Tasks:**
- `filter_groups.metric: "seq_final_reward"` - Use final processed rewards
- `max_num_gen_batches: 5-10` - Allow more generation attempts
- `n_resp_per_prompt: 8-16` - Adjust based on computational budget

## Algorithm Details

### GSPO Loss Function

The GSPO loss uses a sequence-level importance ratio with dual clipping:

```python
# Sequence-level importance ratio
seq_importance_ratio = exp(log_prob - log_prob.detach() + kl_seq.detach())

# Dual clipping
pg_losses1 = -advantages * seq_importance_ratio  
pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1.0 - clip_low, 1.0 + clip_high)

# Final loss
pg_loss = torch.max(pg_losses1, pg_losses2)
```

### Dynamic Sampling Process

1. **Generation**: Generate multiple responses per prompt (`n_resp_per_prompt`)
2. **Reward Computation**: Calculate rewards for all responses
3. **Variance Filtering**: Keep prompts where response rewards have variance > 0
4. **Batch Completion**: If filtered samples < target batch size, generate more batches
5. **Training**: Train on filtered high-quality samples

### Benefits of the Combination

1. **Improved Data Efficiency**: Dynamic sampling focuses training on challenging examples
2. **Better Convergence**: GSPO's dual clipping prevents policy collapse
3. **Flexible Quality Control**: Configurable filtering metrics for different domains
4. **Automatic Adaptation**: System automatically adjusts generation based on data quality

## Performance Tips

### Memory Optimization
- Use `use_dynamic_bsz=true` for automatic batch size adjustment
- Enable `entropy_checkpointing=true` to save memory
- Adjust `ppo_micro_batch_size_per_gpu` based on GPU memory

### Training Speed
- Use `rollout_mode=sync` for smaller models (< 7B)  
- Use `rollout_mode=async` for larger models (>= 7B)
- Increase `gpu_memory_utilization` to 0.8-0.9 for better throughput

### Data Quality
- Start with `max_num_gen_batches=3` and increase if needed
- Monitor `train/num_gen_batches` metric in logs
- Adjust filtering metric based on reward function characteristics

## Monitoring and Debugging

### Key Metrics to Watch

```python
# Dynamic sampling effectiveness
"train/num_gen_batches"          # Average generation batches per training step
"train/filtered_prompts_ratio"   # Ratio of prompts kept after filtering

# GSPO-specific metrics  
"actor/pg_clipfrac_lower"        # Lower clipping fraction
"actor/pg_clipfrac_upper"        # Upper clipping fraction
"actor/seq_importance_ratio"     # Average sequence importance ratio
```

### Common Issues and Solutions

**Issue**: Too many generation batches (> max_num_gen_batches)
- **Solution**: Reduce task difficulty or increase `clip_ratio_low/high`

**Issue**: Low filtering ratio (< 30%)
- **Solution**: Decrease `n_resp_per_prompt` or change filtering metric

**Issue**: Training instability  
- **Solution**: Ensure `clip_ratio_low < clip_ratio_high` and use proper aggregation mode

## File Structure

```
recipe/gspo/
├── README_dynamic_sampling.md          # This documentation
├── gspo_dynamic_sampling_3b_math.sh    # Bash training script
├── main_gspo_dynamic.py                # Python training script  
├── config/
│   └── gspo_dynamic_sampling.yaml      # Configuration file
└── test_gspo_3b_math.sh                # Original GSPO script (reference)
```

## References

- [GSPO Paper](https://arxiv.org/pdf/2507.18071) - Group Self-Play Optimization
- [DAPO Implementation](../../dapo/) - Dynamic sampling mechanism
- [verl Documentation](https://verl.readthedocs.io/) - Framework documentation