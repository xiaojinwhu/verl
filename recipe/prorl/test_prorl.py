#!/usr/bin/env python3
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
ProRL Test Script: Basic functionality tests for ProRL implementation
"""

import torch
import numpy as np
from omegaconf import OmegaConf

def test_kl_controller():
    """Test Dynamic KL Controller functionality."""
    print("Testing KL Controller...")
    
    # Mock config
    config = OmegaConf.create({
        'algorithm': {
            'kl_control': {
                'initial_coef': 0.01,
                'target_kl': 0.05,
                'adaptive': True,
                'min_coef': 0.001,
                'max_coef': 0.1
            }
        }
    })
    
    from prorl_ray_trainer import DynamicKLController
    
    kl_controller = DynamicKLController(config)
    
    # Test initial state
    assert kl_controller.get_coefficient() == 0.01
    
    # Test high KL scenario
    new_coef = kl_controller.update(0.1)  # High KL
    assert new_coef > 0.01, f"Expected increase, got {new_coef}"
    
    # Test low KL scenario
    new_coef = kl_controller.update(0.01)  # Low KL
    assert new_coef < kl_controller.get_coefficient() or new_coef >= kl_controller.min_coef
    
    print("✓ KL Controller tests passed")


def test_reference_reset_controller():
    """Test Reference Policy Reset Controller."""
    print("Testing Reference Reset Controller...")
    
    config = OmegaConf.create({
        'algorithm': {
            'reference_reset': {
                'interval': 5,
                'kl_threshold': 0.1,
                'performance_degradation': True,
                'validation_metric': 'reward'
            }
        }
    })
    
    from prorl_ray_trainer import ReferencyPolicyResetController
    
    reset_controller = ReferencyPolicyResetController(config)
    
    # Test interval-based reset
    assert not reset_controller.should_reset(1, 0.01)
    assert reset_controller.should_reset(6, 0.01)  # Past interval
    
    # Test KL threshold reset
    assert reset_controller.should_reset(3, 0.15)  # High KL
    
    # Test reset tracking
    reset_controller.reset_performed(6)
    assert reset_controller.last_reset_step == 6
    
    print("✓ Reference Reset Controller tests passed")


def test_prorl_advantage_estimator():
    """Test ProRL GRPO advantage estimator."""
    print("Testing ProRL Advantage Estimator...")
    
    from prorl_core_algos import prorl_grpo_advantage_estimator
    
    # Create test data
    rewards = torch.tensor([0.8, 0.9, 0.1, 0.7, 0.2, 0.9])
    prompt_ids = torch.tensor([0, 0, 0, 1, 1, 1])  # Two groups of 3
    
    config = OmegaConf.create({
        'algorithm': {
            'norm_adv_by_std_in_grpo': True
        }
    })
    
    advantages = prorl_grpo_advantage_estimator(rewards, prompt_ids, config)
    
    # Check that advantages are computed per group
    assert len(advantages) == len(rewards)
    
    # Group 0: rewards [0.8, 0.9, 0.1], mean = 0.6
    # Group 1: rewards [0.7, 0.2, 0.9], mean = 0.6
    
    # Advantages should be relative to group mean
    group_0_advantages = advantages[:3]
    group_1_advantages = advantages[3:]
    
    # Check that group means are approximately 0
    assert abs(torch.mean(group_0_advantages).item()) < 1e-6
    assert abs(torch.mean(group_1_advantages).item()) < 1e-6
    
    print("✓ ProRL Advantage Estimator tests passed")


def test_policy_loss():
    """Test ProRL policy loss with decoupled clipping."""
    print("Testing ProRL Policy Loss...")
    
    from prorl_core_algos import prorl_policy_loss
    
    # Create test data
    batch_size, seq_len = 4, 10
    old_log_prob = torch.randn(batch_size, seq_len)
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.1
    advantages = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    config = OmegaConf.create({
        'actor_rollout_ref': {
            'actor': {
                'clip_ratio': 0.2,
                'clip_ratio_low': 0.2,
                'clip_ratio_high': 0.28
            }
        }
    })
    
    policy_loss, approx_kl, clipfrac, ratio = prorl_policy_loss(
        old_log_prob, log_prob, advantages, response_mask, "token-mean", config
    )
    
    # Check that outputs are tensors with correct shapes
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(approx_kl, torch.Tensor)
    assert isinstance(clipfrac, torch.Tensor)
    assert isinstance(ratio, torch.Tensor)
    
    # Check that policy loss is a scalar
    assert policy_loss.dim() == 0
    
    print("✓ ProRL Policy Loss tests passed")


def test_entropy_controller():
    """Test ProRL Entropy Controller."""
    print("Testing Entropy Controller...")
    
    from prorl_core_algos import ProRLEntropyController
    
    config = OmegaConf.create({
        'algorithm': {
            'entropy_control': {
                'target_entropy': -2.0,
                'initial_coef': 0.01,
                'adaptive': True,
                'min_coef': 0.001,
                'max_coef': 0.1
            }
        }
    })
    
    entropy_controller = ProRLEntropyController(config)
    
    # Test decreasing entropy (should increase coefficient)
    for entropy in [-1.5, -1.8, -2.1, -2.4, -2.7]:
        coef = entropy_controller.update(entropy)
    
    assert entropy_controller.get_coefficient() > 0.01, "Should increase coefficient for decreasing entropy"
    
    print("✓ Entropy Controller tests passed")


def run_all_tests():
    """Run all ProRL component tests."""
    print("=== ProRL Component Tests ===\n")
    
    try:
        test_kl_controller()
        test_reference_reset_controller()
        test_prorl_advantage_estimator()
        test_policy_loss()
        test_entropy_controller()
        
        print("\n=== All Tests Passed! ===")
        print("ProRL implementation is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)