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
ProRL Ray Trainer: Implements Prolonged Reinforcement Learning with KL divergence control,
reference policy resetting, and DAPO-style enhancements.

Based on the paper: "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries 
in Large Language Models"
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import random
import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    Role,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class ReferencyPolicyResetController:
    """Controls when and how to reset the reference policy during ProRL training."""
    
    def __init__(self, config):
        self.config = config
        self.reset_config = config.algorithm.get('reference_reset', {})
        
        # Reset triggers
        self.reset_interval = self.reset_config.get('interval', 100)  # Reset every N steps
        self.kl_threshold = self.reset_config.get('kl_threshold', 0.1)  # Reset if KL exceeds this
        self.performance_degradation = self.reset_config.get('performance_degradation', True)
        self.validation_metric = self.reset_config.get('validation_metric', 'reward')
        
        # Reset tracking
        self.last_reset_step = 0
        self.validation_history = []
        self.best_validation = float('-inf')
        
    def should_reset(self, step: int, kl_div: float, validation_metrics: dict = None) -> bool:
        """Determine if reference policy should be reset."""
        
        # Check interval-based reset
        if step - self.last_reset_step >= self.reset_interval:
            return True
            
        # Check KL divergence threshold
        if kl_div > self.kl_threshold:
            return True
            
        # Check performance degradation
        if validation_metrics and self.performance_degradation:
            current_metric = validation_metrics.get(self.validation_metric, float('-inf'))
            self.validation_history.append(current_metric)
            
            # Keep only recent history
            if len(self.validation_history) > 10:
                self.validation_history = self.validation_history[-10:]
                
            # Check for degradation (no improvement in last 5 validations)
            if len(self.validation_history) >= 5:
                recent_max = max(self.validation_history[-5:])
                if recent_max < self.best_validation * 0.95:  # 5% degradation threshold
                    return True
                    
            if current_metric > self.best_validation:
                self.best_validation = current_metric
                
        return False
        
    def reset_performed(self, step: int):
        """Call this when a reset has been performed."""
        self.last_reset_step = step
        self.validation_history = []


class DynamicKLController:
    """Enhanced KL controller for ProRL with adaptive penalties."""
    
    def __init__(self, config):
        self.config = config
        kl_config = config.algorithm.get('kl_control', {})
        
        self.initial_coef = kl_config.get('initial_coef', 0.01)
        self.target_kl = kl_config.get('target_kl', 0.05)
        self.adaptive = kl_config.get('adaptive', True)
        self.min_coef = kl_config.get('min_coef', 0.001)
        self.max_coef = kl_config.get('max_coef', 0.1)
        
        self.current_coef = self.initial_coef
        self.history = []
        
    def update(self, kl_div: float) -> float:
        """Update KL coefficient based on observed KL divergence."""
        if not self.adaptive:
            return self.current_coef
            
        self.history.append(kl_div)
        if len(self.history) > 10:
            self.history = self.history[-10:]
            
        # Adaptive adjustment
        if kl_div > self.target_kl * 1.5:
            # KL too high, increase penalty
            self.current_coef = min(self.current_coef * 1.2, self.max_coef)
        elif kl_div < self.target_kl * 0.5:
            # KL too low, decrease penalty
            self.current_coef = max(self.current_coef * 0.9, self.min_coef)
            
        return self.current_coef
        
    def get_coefficient(self) -> float:
        return self.current_coef


class RayProRLTrainer(RayPPOTrainer):
    """
    ProRL Trainer: Implements Prolonged Reinforcement Learning with enhanced GRPO,
    KL divergence control, and reference policy resetting.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize ProRL components
        self.reference_reset_controller = ReferencyPolicyResetController(self.config)
        self.kl_controller = DynamicKLController(self.config)
        
        # DAPO-style enhancements
        self.enable_dynamic_sampling = self.config.algorithm.get('filter_groups', {}).get('enable', False)
        self.enable_clip_higher = getattr(self.config.actor_rollout_ref.actor, 'clip_ratio_high', None) is not None
        
        # ProRL specific settings
        self.prolonged_training = self.config.algorithm.get('prolonged_training', {})
        self.max_response_length = self.config.data.get('max_response_length', 8192)
        
        # Tracking variables
        self.reset_count = 0
        self.entropy_history = []
        
    def fit(self):
        """
        The training loop of ProRL.
        Extends the standard PPO training loop with ProRL-specific enhancements.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # Add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="ProRL Training")

        # Start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.config.global_profiler.tool is not None
            and self.global_steps in self.config.global_profiler.get("steps", [])
        )

        # Main training loop
        while self.global_steps <= self.total_training_steps:
            try:
                # Profile management
                if not prev_step_profile and curr_step_profile:
                    self._start_profile()
                if prev_step_profile and not curr_step_profile:
                    self._stop_profile()

                prev_step_profile = curr_step_profile

                # Generate rollouts with dynamic sampling if enabled
                batch = self._generate_rollouts_with_dynamic_sampling()
                
                # Compute advantages using GRPO or specified estimator
                batch = self._compute_advantages(batch)
                
                # Apply KL divergence control
                kl_div = self._apply_kl_control(batch)
                
                # Check if reference policy should be reset
                if self.reference_reset_controller.should_reset(
                    self.global_steps, kl_div, last_val_metrics
                ):
                    self._perform_reference_reset()
                
                # Update actor using ProRL enhancements
                with marked_timer() as timer:
                    actor_metrics = self._update_actor_with_prorl(batch)
                    
                compute_timing_metrics.extend_actor_update_time(timer.elapse)

                # Compute metrics
                metrics = {}
                data_metrics = compute_data_metrics(batch)
                timing_metrics = compute_timing_metrics(self.actor_rollout_wg.world_size)
                throughput_metrics = compute_throughout_metrics(
                    timing_metrics, data_metrics, self.config.trainer.nnodes
                )
                
                # Add ProRL specific metrics
                prorl_metrics = self._compute_prorl_metrics(batch, kl_div)
                
                metrics.update(data_metrics)
                metrics.update(timing_metrics)
                metrics.update(throughput_metrics)
                metrics.update(actor_metrics)
                metrics.update(prorl_metrics)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.global_steps % self.config.trainer.val_freq == 0
                ):
                    val_metrics = self._validate()
                    last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Save checkpoint
                if self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()

                # Log and update
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.set_description(
                    f"ProRL Training - Step {self.global_steps}, Reward: {metrics.get('reward/mean', 0):.3f}, "
                    f"KL: {kl_div:.4f}, Resets: {self.reset_count}"
                )
                progress_bar.update(1)

                # Prepare for next step
                compute_timing_metrics.reset()
                self.global_steps += 1
                self.gen_steps += 1

                # Update next step profiling
                curr_step_profile = (
                    self.config.global_profiler.tool is not None
                    and self.global_steps in self.config.global_profiler.get("steps", [])
                )

            except Exception as e:
                print(f"Error in training step {self.global_steps}: {e}")
                raise

        progress_bar.close()

        # Final checkpoint save
        self._save_checkpoint()
        if prev_step_profile:
            self._stop_profile()
            
    def _generate_rollouts_with_dynamic_sampling(self):
        """Generate rollouts with DAPO-style dynamic sampling if enabled."""
        if not self.enable_dynamic_sampling:
            return self._generate_sequences()
            
        # Dynamic sampling implementation
        batches = []
        num_prompt_in_batch = 0
        prompt_bsz = self.config.data.train_batch_size
        num_gen_batches = 0
        max_gen_batches = self.config.algorithm.filter_groups.get('max_num_gen_batches', 10)
        
        while True:
            batch = self._generate_sequences()
            batches.append(batch)
            
            # Filter groups with all same rewards (0 or 1)
            filtered_batch = self._filter_groups(batch)
            num_prompt_in_batch += len(filtered_batch.prompts)
            
            if num_prompt_in_batch >= prompt_bsz:
                # Combine and align batches
                combined_batch = self._combine_batches(batches)
                traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                return combined_batch[:traj_bsz]
                
            num_gen_batches += 1
            if max_gen_batches > 0 and num_gen_batches >= max_gen_batches:
                print(f"Reached max generation batches ({max_gen_batches}), using available data")
                return self._combine_batches(batches)
                
    def _filter_groups(self, batch):
        """Filter out groups where all outputs have the same reward (0 or 1)."""
        if not self.enable_dynamic_sampling:
            return batch
            
        # Group responses by prompt
        grouped_responses = defaultdict(list)
        for i, prompt_id in enumerate(batch.prompt_ids):
            grouped_responses[prompt_id].append(i)
            
        # Filter groups
        valid_indices = []
        for prompt_id, indices in grouped_responses.items():
            rewards = [batch.rewards[i] for i in indices]
            # Keep group if rewards are not all 0 or all 1
            if not (all(r == 0 for r in rewards) or all(r == 1 for r in rewards)):
                valid_indices.extend(indices)
                
        # Return filtered batch
        if valid_indices:
            return batch.select(valid_indices)
        else:
            return batch  # Return original if no valid groups
            
    def _apply_kl_control(self, batch):
        """Apply dynamic KL divergence control."""
        # Compute current KL divergence
        if hasattr(batch, 'ref_log_probs') and hasattr(batch, 'log_probs'):
            kl_div = torch.mean(batch.log_probs - batch.ref_log_probs).item()
        else:
            kl_div = 0.0
            
        # Update KL controller
        kl_coef = self.kl_controller.update(kl_div)
        
        # Apply KL penalty to rewards
        if self.config.algorithm.get('use_kl_in_reward', True):
            batch = apply_kl_penalty(batch, kl_coef)
            
        return kl_div
        
    def _perform_reference_reset(self):
        """Perform hard reset of reference policy and optimizer."""
        print(f"Performing reference policy reset at step {self.global_steps}")
        
        # Reset reference policy to current actor policy
        if hasattr(self, 'ref_policy_wg'):
            # Copy actor weights to reference policy
            actor_state = self.actor_rollout_wg.get_checkpoint()
            self.ref_policy_wg.load_checkpoint(actor_state)
            
        # Reset optimizer states
        self.actor_rollout_wg.reset_optimizer()
        
        # Update controllers
        self.reference_reset_controller.reset_performed(self.global_steps)
        self.reset_count += 1
        
        # Reset KL controller
        self.kl_controller.current_coef = self.kl_controller.initial_coef
        
    def _update_actor_with_prorl(self, batch):
        """Update actor with ProRL enhancements including DAPO-style clipping."""
        # Standard actor update with ProRL modifications
        if self.enable_clip_higher:
            # Use DAPO-style decoupled clipping
            clip_low = self.config.actor_rollout_ref.actor.get('clip_ratio_low', 0.2)
            clip_high = self.config.actor_rollout_ref.actor.get('clip_ratio_high', 0.28)
            
            # Modify batch to use decoupled clipping
            batch.clip_ratio_low = clip_low
            batch.clip_ratio_high = clip_high
            
        return self.actor_rollout_wg.update_policy(batch)
        
    def _compute_prorl_metrics(self, batch, kl_div):
        """Compute ProRL-specific metrics."""
        metrics = {
            'prorl/kl_divergence': kl_div,
            'prorl/kl_coefficient': self.kl_controller.get_coefficient(),
            'prorl/reset_count': self.reset_count,
            'prorl/steps_since_reset': self.global_steps - self.reference_reset_controller.last_reset_step,
        }
        
        # Entropy tracking
        if hasattr(batch, 'log_probs'):
            entropy = -torch.mean(batch.log_probs).item()
            self.entropy_history.append(entropy)
            if len(self.entropy_history) > 100:
                self.entropy_history = self.entropy_history[-100:]
                
            metrics['prorl/entropy'] = entropy
            metrics['prorl/entropy_trend'] = (
                np.mean(self.entropy_history[-10:]) - np.mean(self.entropy_history[-20:-10])
                if len(self.entropy_history) >= 20 else 0.0
            )
            
        # Response length metrics
        if hasattr(batch, 'response_length'):
            metrics['prorl/avg_response_length'] = torch.mean(batch.response_length.float()).item()
            metrics['prorl/max_response_length'] = torch.max(batch.response_length).item()
            
        return metrics
        
    def _combine_batches(self, batches):
        """Combine multiple batches into one."""
        if len(batches) == 1:
            return batches[0]
            
        # Combine all attributes
        combined = DataProto()
        for attr in dir(batches[0]):
            if not attr.startswith('_') and hasattr(batches[0], attr):
                values = []
                for batch in batches:
                    if hasattr(batch, attr):
                        values.append(getattr(batch, attr))
                        
                if values and torch.is_tensor(values[0]):
                    setattr(combined, attr, torch.cat(values, dim=0))
                elif values and isinstance(values[0], list):
                    combined_list = []
                    for v in values:
                        combined_list.extend(v)
                    setattr(combined, attr, combined_list)
                    
        return combine
    
    
    d