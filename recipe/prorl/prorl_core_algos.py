# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ProRL Core Algorithms: Enhanced policy loss functions and advantage estimators
for Prolonged Reinforcement Learning with DAPO-style improvements.
"""

import torch
from typing import Optional
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss, register_adv_est
from verl.trainer.config import AlgoConfig


@register_policy_loss("prorl_clip")
def prorl_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ProRL policy loss with DAPO-style decoupled clipping for enhanced exploration.
    
    Implements "clip-higher" strategy where upper clip bound is higher to encourage
    exploration while maintaining stability.
    """
    ratio = torch.exp(log_prob - old_log_prob)
    
    # Get clipping parameters - support both regular and decoupled clipping
    if config and hasattr(config, 'actor_rollout_ref'):
        actor_config = config.actor_rollout_ref.actor
        clip_ratio_low = getattr(actor_config, 'clip_ratio_low', actor_config.clip_ratio)
        clip_ratio_high = getattr(actor_config, 'clip_ratio_high', actor_config.clip_ratio)
    else:
        clip_ratio_low = clip_ratio_high = 0.2
    
    # ProRL: Decoupled clipping for better exploration
    clip_low = 1.0 - clip_ratio_low
    clip_high = 1.0 + clip_ratio_high
    
    # Compute policy losses
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, clip_low, clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    # Apply response mask and aggregate
    pg_losses = pg_losses * response_mask
    
    if loss_agg_mode == "token-mean":
        policy_loss = verl_F.masked_mean(pg_losses, response_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(pg_losses, dim=-1)
        policy_loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_lengths = torch.sum(response_mask, dim=-1)
        seq_losses = torch.sum(pg_losses, dim=-1) / (seq_lengths + 1e-8)
        policy_loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # DrGRPO style normalization
        seq_losses = torch.sum(pg_losses, dim=-1)
        policy_loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    
    # Compute additional metrics
    approx_kl = verl_F.masked_mean((log_prob - old_log_prob), response_mask)
    clipfrac = verl_F.masked_mean(
        ((ratio - 1.0).abs() > clip_ratio_high).float(), response_mask
    )
    
    return policy_loss, approx_kl, clipfrac, ratio


@register_adv_est("prorl_grpo")
def prorl_grpo_advantage_estimator(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor, 
    config: Optional[DictConfig | AlgoConfig] = None,
) -> torch.Tensor:
    """
    ProRL GRPO advantage estimator with enhanced group-based normalization.
    
    Implements group relative policy optimization with optional improvements
    for prolonged training stability.
    """
    advantages = torch.zeros_like(rewards)
    
    # Group responses by prompt ID
    unique_prompt_ids = torch.unique(prompt_ids)
    
    for prompt_id in unique_prompt_ids:
        group_mask = (prompt_ids == prompt_id)
        group_rewards = rewards[group_mask]
        
        if len(group_rewards) <= 1:
            # Single response, no relative comparison
            advantages[group_mask] = 0.0
            continue
            
        # Compute group statistics
        group_mean = torch.mean(group_rewards)
        group_std = torch.std(group_rewards)
        
        # ProRL: Enhanced normalization for prolonged training
        if config and hasattr(config, 'algorithm'):
            algo_config = config.algorithm
            norm_by_std = algo_config.get('norm_adv_by_std_in_grpo', True)
            
            if norm_by_std and group_std > 1e-8:
                group_advantages = (group_rewards - group_mean) / (group_std + 1e-8)
            else:
                group_advantages = group_rewards - group_mean
        else:
            # Standard GRPO normalization
            if group_std > 1e-8:
                group_advantages = (group_rewards - group_mean) / (group_std + 1e-8)
            else:
                group_advantages = group_rewards - group_mean
        
        advantages[group_mask] = group_advantages
    
    return advantages


def apply_prorl_kl_penalty(
    batch,
    kl_coef: float,
    ref_log_probs: Optional[torch.Tensor] = None
):
    """
    Apply ProRL-style KL penalty with dynamic coefficient adjustment.
    
    Args:
        batch: Training batch containing log probabilities
        kl_coef: KL penalty coefficient
        ref_log_probs: Reference policy log probabilities
    """
    if ref_log_probs is None and hasattr(batch, 'ref_log_probs'):
        ref_log_probs = batch.ref_log_probs
        
    if ref_log_probs is not None and hasattr(batch, 'log_probs'):
        # Compute KL divergence
        kl_penalty = kl_coef * (batch.log_probs - ref_log_probs)
        
        # Apply penalty to rewards
        if hasattr(batch, 'rewards'):
            batch.rewards = batch.rewards - kl_penalty.sum(dim=-1)
            
    return batch


class ProRLEntropyController:
    """
    Entropy controller for ProRL to prevent entropy collapse during prolonged training.
    
    Implements dynamic entropy regularization based on training progress and entropy trends.
    """
    
    def __init__(self, config):
        self.config = config
        entropy_config = config.algorithm.get('entropy_control', {})
        
        self.target_entropy = entropy_config.get('target_entropy', -2.0)
        self.entropy_coef = entropy_config.get('initial_coef', 0.01)
        self.adaptive = entropy_config.get('adaptive', True)
        self.min_coef = entropy_config.get('min_coef', 0.001)
        self.max_coef = entropy_config.get('max_coef', 0.1)
        
        self.entropy_history = []
        
    def update(self, current_entropy: float) -> float:
        """Update entropy coefficient based on current entropy."""
        if not self.adaptive:
            return self.entropy_coef
            
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > 20:
            self.entropy_history = self.entropy_history[-20:]
            
        # Check for entropy collapse
        if len(self.entropy_history) >= 10:
            recent_trend = (
                sum(self.entropy_history[-5:]) / 5 - 
                sum(self.entropy_history[-10:-5]) / 5
            )
            
            if recent_trend < -0.1:  # Entropy decreasing
                self.entropy_coef = min(self.entropy_coef * 1.2, self.max_coef)
            elif recent_trend > 0.05:  # Entropy increasing too much
                self.entropy_coef = max(self.entropy_coef * 0.9, self.min_coef)
                
        return self.entropy_coef
        
    def get_coefficient(self) -> float:
        return self.entropy_coef