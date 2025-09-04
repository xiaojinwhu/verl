# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient and production-ready RL training library for large language models (LLMs). It's the open-source version of the HybridFlow framework designed for post-training LLMs with reinforcement learning.

## Development Commands

### Installation and Setup
```bash
# Install base dependencies
pip install -r requirements.txt

# Install with specific backends
pip install -e ".[vllm]"     # For vLLM backend
pip install -e ".[sglang]"   # For SGLang backend
pip install -e ".[gpu]"      # For GPU-specific kernels (flash-attn, liger-kernel)
pip install -e ".[test]"     # For testing dependencies
```

### Testing
```bash
# Run basic CPU tests
python -m pytest tests/ -k "on_cpu"

# Run specific test categories
python -m pytest tests/special_sanity/  # Sanity checks
python -m pytest tests/utils/          # Utility tests
python -m pytest tests/single_controller/  # Controller tests
```

### Code Quality
```bash
# Run linting and formatting
ruff check verl/ --fix
ruff format verl/

# Run type checking (limited coverage due to config)
mypy verl/trainer/config/algorithm.py
mypy verl/trainer/ppo/core_algos.py
mypy verl/workers/reward_manager/
```

### Example Training Runs
```bash
# GRPO training with Qwen2-7B
bash examples/grpo_trainer/run_qwen2-7b.sh

# PPO training with math rewards
bash examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh

# Multi-turn training with SGLang
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh

# Supervised fine-tuning
bash examples/sft/gsm8k/run_qwen_05_sp2.sh
```

## Architecture Overview

### Core Components

**Hybrid Controller Architecture**: The framework uses a hybrid-controller programming model that decouples computation and data dependencies, enabling flexible RL dataflow execution.

**Multi-Backend Support**:
- **Training Backends**: FSDP, FSDP2, Megatron-LM
- **Inference Backends**: vLLM, SGLang, HuggingFace Transformers
- **Hardware**: CUDA GPUs, NPU (Ascend), AMD ROCm

**Key Modules**:
- `verl/trainer/`: Main training orchestration and algorithms (PPO, GRPO, etc.)
- `verl/workers/`: Distributed workers for actor, critic, reward models
- `verl/single_controller/`: Ray-based worker management and coordination
- `verl/models/`: Model implementations for different backends (Transformers, Megatron)
- `verl/utils/`: Utilities for datasets, checkpointing, debugging, device management

### Supported Algorithms
- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization) 
- **GSPO**, **ReMax**, **REINFORCE++**, **RLOO**, **PRIME**
- **DAPO**, **DrGRPO**, **PF-PPO** (advanced variants)

### Model Support
- Qwen (2, 2.5, 3), Qwen-VL series for multi-modal RL
- Llama 3.1, Gemma2, DeepSeek-LLM
- Large MoE models (DeepSeek-671B, Qwen3-236B) via Megatron backend
- Custom model integration through registries

## Configuration System

The project uses Hydra for configuration management with YAML config files:

**Main config locations**:
- `verl/trainer/config/`: Core training configurations
- `examples/*/config/`: Algorithm-specific example configs
- Runtime configs passed via command line or config overrides

**Key config patterns**:
```yaml
# Training backend selection
actor_rollout_ref.actor.strategy: fsdp2
critic.strategy: fsdp2
reward_model.strategy: fsdp2

# Rollout backend selection  
rollout.name: vllm  # or sglang, hf
rollout.tensor_model_parallel_size: 1
```

## Development Patterns

**Model Registration**: New models are registered via `verl/models/registry.py` and backend-specific modules.

**Worker Architecture**: Uses Ray for distributed coordination with workers implementing specific roles (actor, critic, rollout, reward).

**Data Flow**: RL training follows generate → reward → train cycles with configurable placement strategies for efficient GPU utilization.

**Memory Optimization**: Supports various memory-saving techniques including LoRA, gradient checkpointing, CPU offloading, and sequence parallelism.

## Testing Strategy

The test suite is organized by component and execution environment:
- `tests/*_on_cpu.py`: CPU-only tests for CI
- `tests/special_distributed/`: Multi-GPU distributed tests  
- `tests/special_e2e/`: End-to-end integration tests
- `tests/special_sanity/`: Code quality and import tests

## Important Notes

- Always check example scripts in `examples/` for reference implementations
- The framework supports both synchronous and asynchronous rollout modes
- Multi-modal RL is supported through VLM integration (Qwen2.5-VL)
- Tool-calling and multi-turn conversations are supported via SGLang backend
- Performance tuning is critical - refer to the performance tuning guide in docs