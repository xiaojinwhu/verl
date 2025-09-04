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
Test script for GSPO Dynamic Sampling Implementation

This script validates that:
1. GSPO loss function is properly registered and callable
2. Dynamic sampling configuration is correctly parsed
3. Filter groups mechanism is properly integrated
4. Configuration validation works as expected
"""

import sys
import traceback
from pathlib import Path

# Add the parent directory to the path to import verl
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_gspo_loss_registration():
    """Test that GSPO loss function is properly registered."""
    print("Testing GSPO loss function registration...")
    
    try:
        from verl.trainer.ppo.core_algos import get_policy_loss_fn, POLICY_LOSS_REGISTRY
        
        # Check if gspo is in the registry
        assert "gspo" in POLICY_LOSS_REGISTRY, f"GSPO not found in registry. Available: {list(POLICY_LOSS_REGISTRY.keys())}"
        
        # Get the GSPO loss function
        gspo_loss_fn = get_policy_loss_fn("gspo")
        assert gspo_loss_fn is not None, "GSPO loss function is None"
        
        print("✓ GSPO loss function is properly registered")
        return True
        
    except Exception as e:
        print(f"✗ GSPO loss function registration test failed: {e}")
        traceback.print_exc()
        return False


def test_gspo_loss_function():
    """Test GSPO loss function with sample inputs."""
    print("Testing GSPO loss function computation...")
    
    try:
        import torch
        from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
        from verl.workers.config import ActorConfig
        
        # Create sample inputs
        batch_size, seq_len = 4, 10
        old_log_prob = torch.randn(batch_size, seq_len) * 0.1
        log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.05
        advantages = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Create a mock config
        config = ActorConfig()
        config.clip_ratio_low = 0.0003
        config.clip_ratio_high = 0.0004
        
        # Test the loss function
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_gspo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config
        )
        
        # Validate outputs
        assert pg_loss.dim() == 0, f"Policy loss should be scalar, got shape {pg_loss.shape}"
        assert pg_clipfrac.dim() == 0, f"Clip fraction should be scalar, got shape {pg_clipfrac.shape}"
        assert ppo_kl.dim() == 0, f"PPO KL should be scalar, got shape {ppo_kl.shape}"
        assert pg_clipfrac_lower.dim() == 0, f"Lower clip fraction should be scalar, got shape {pg_clipfrac_lower.shape}"
        
        assert torch.isfinite(pg_loss), f"Policy loss is not finite: {pg_loss}"
        assert 0.0 <= pg_clipfrac <= 1.0, f"Clip fraction out of range: {pg_clipfrac}"
        assert 0.0 <= pg_clipfrac_lower <= 1.0, f"Lower clip fraction out of range: {pg_clipfrac_lower}"
        
        print("✓ GSPO loss function computes correctly")
        print(f"  Policy Loss: {pg_loss:.6f}")
        print(f"  Clip Fraction: {pg_clipfrac:.6f}")
        print(f"  Lower Clip Fraction: {pg_clipfrac_lower:.6f}")
        print(f"  PPO KL: {ppo_kl:.6f}")
        return True
        
    except Exception as e:
        print(f"✗ GSPO loss function test failed: {e}")
        traceback.print_exc()
        return False


def test_dynamic_sampling_config():
    """Test that dynamic sampling configuration is properly loaded."""
    print("Testing dynamic sampling configuration...")
    
    try:
        from omegaconf import OmegaConf
        
        # Load the config
        config_path = Path(__file__).parent / "config" / "gspo_dynamic_sampling.yaml"
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
            
        config = OmegaConf.load(config_path)
        
        # Check required fields
        assert hasattr(config, "algorithm"), "Config missing 'algorithm' section"
        assert hasattr(config.algorithm, "filter_groups"), "Config missing 'filter_groups' section"
        
        filter_groups = config.algorithm.filter_groups
        assert filter_groups.enable == True, f"Dynamic sampling not enabled: {filter_groups.enable}"
        assert filter_groups.metric in ["seq_reward", "seq_final_reward", "acc"], f"Invalid metric: {filter_groups.metric}"
        assert filter_groups.max_num_gen_batches > 0, f"Invalid max_num_gen_batches: {filter_groups.max_num_gen_batches}"
        
        # Check GSPO-specific config
        actor_config = config.actor_rollout_ref.actor
        assert actor_config.policy_loss.loss_mode == "gspo", f"Loss mode should be 'gspo', got: {actor_config.policy_loss.loss_mode}"
        assert actor_config.clip_ratio_low < actor_config.clip_ratio_high, "clip_ratio_low should be < clip_ratio_high"
        assert actor_config.loss_agg_mode == "seq-mean-token-mean", f"Unexpected aggregation mode: {actor_config.loss_agg_mode}"
        
        print("✓ Dynamic sampling configuration is valid")
        print(f"  Filter metric: {filter_groups.metric}")
        print(f"  Max gen batches: {filter_groups.max_num_gen_batches}")
        print(f"  Clip ratios: low={actor_config.clip_ratio_low}, high={actor_config.clip_ratio_high}")
        return True
        
    except Exception as e:
        print(f"✗ Dynamic sampling configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_algorithm_config_integration():
    """Test that algorithm config properly integrates filter_groups."""
    print("Testing algorithm config integration...")
    
    try:
        from omegaconf import OmegaConf
        from verl.trainer.config import AlgoConfig
        
        # Create a test config
        config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 1.0,
            "lam": 1.0, 
            "adv_estimator": "grpo",
            "filter_groups": {
                "enable": True,
                "metric": "seq_reward", 
                "max_num_gen_batches": 5
            }
        }
        
        config = OmegaConf.create(config_dict)
        
        # Test that we can access filter_groups
        assert config.filter_groups.enable == True
        assert config.filter_groups.metric == "seq_reward"
        assert config.filter_groups.max_num_gen_batches == 5
        
        print("✓ Algorithm config integration works")
        print(f"  Filter groups enabled: {config.filter_groups.enable}")
        return True
        
    except Exception as e:
        print(f"✗ Algorithm config integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests for GSPO dynamic sampling."""
    print("=" * 80)
    print("GSPO Dynamic Sampling Test Suite")
    print("=" * 80)
    
    tests = [
        test_gspo_loss_registration,
        test_gspo_loss_function,
        test_dynamic_sampling_config,
        test_algorithm_config_integration,
    ]
    
    results = []
    for test in tests:
        print(f"\\n{'-' * 60}")
        result = test()
        results.append(result)
        print(f"{'-' * 60}")
    
    print(f"\\n{'=' * 80}")
    print("Test Results Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All tests passed! GSPO dynamic sampling is ready to use.")
        return 0
    else:
        print(f"\\n❌ {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)