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
GSPO Training with Dynamic Sampling

This script demonstrates how to integrate DAPO's dynamic sampling mechanism 
with the GSPO (Group Self-Play Optimization) algorithm. The dynamic sampling
helps improve data efficiency by filtering high-quality training examples
based on reward variance.

Key Features:
1. GSPO algorithm with dual clipping ratios
2. Dynamic sampling based on reward variance  
3. Configurable filtering metrics (seq_reward, seq_final_reward, acc)
4. Automatic batch size adjustment for filtered samples
"""

import os
import socket
import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="gspo_dynamic_sampling", version_base=None)
def main(config):
    """Main entry point for GSPO training with dynamic sampling."""
    print("=" * 80)
    print("GSPO Training with Dynamic Sampling")
    print("=" * 80)
    print(f"Algorithm: GSPO (Group Self-Play Optimization)")
    print(f"Dynamic Sampling: {'Enabled' if config.algorithm.filter_groups.enable else 'Disabled'}")
    if config.algorithm.filter_groups.enable:
        print(f"Filter Metric: {config.algorithm.filter_groups.metric}")
        print(f"Max Generation Batches: {config.algorithm.filter_groups.max_num_gen_batches}")
    print(f"Model: {config.actor_rollout_ref.model.path}")
    print(f"Loss Mode: {config.actor_rollout_ref.actor.policy_loss.loss_mode}")
    print(f"Clip Ratios: Low={config.actor_rollout_ref.actor.clip_ratio_low}, High={config.actor_rollout_ref.actor.clip_ratio_high}")
    print("=" * 80)
    
    run_gspo_training(config)


def run_gspo_training(config) -> None:
    """
    Run GSPO training with dynamic sampling.
    
    The training process includes:
    1. Dynamic sample generation with configurable filtering
    2. GSPO loss computation with dual clipping ratios
    3. Advantage estimation using GRPO
    4. Batch balancing for filtered high-quality samples
    """
    if not ray.is_initialized():
        # Initialize Ray with dynamic sampling friendly settings
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true", 
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN"
            }
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Setup profiling if needed
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote task runner for GSPO dynamic sampling training."""
    
    def run(self, config):
        """Execute GSPO training with dynamic sampling."""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print("\n" + "=" * 50)
        print("GSPO Dynamic Sampling Configuration:")
        print("=" * 50)
        pprint(OmegaConf.to_container(config, resolve=True))
        print("=" * 50)
        
        OmegaConf.resolve(config)

        # Download model from remote if needed
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Setup tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        # Setup workers based on strategy
        from verl.single_controller.ray import RayWorkerGroup
        
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        # Setup resource management
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Setup reward model if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Setup reference model if needed for KL penalty
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Setup reward functions
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        print("\n" + "=" * 50)
        print("Initializing GSPO Trainer with Dynamic Sampling...")
        print("=" * 50)
        
        # Create trainer
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        
        print("Initializing workers...")
        trainer.init_workers()
        
        print("Starting GSPO training with dynamic sampling...")
        trainer.fit()
        
        print("\n" + "=" * 50)
        print("GSPO Training Completed Successfully!")
        print("=" * 50)


if __name__ == "__main__":
    main()