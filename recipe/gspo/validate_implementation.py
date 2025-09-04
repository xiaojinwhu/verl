#!/usr/bin/env python3
"""
Simple validation script for GSPO Dynamic Sampling implementation
This script validates file structure and configuration without requiring dependencies.
"""

import os
import yaml
from pathlib import Path


def validate_file_structure():
    """Validate that all required files are present."""
    print("Validating file structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        "gspo_dynamic_sampling_3b_math.sh",
        "main_gspo_dynamic.py", 
        "config/gspo_dynamic_sampling.yaml",
        "README_dynamic_sampling.md",
        "test_gspo_dynamic_sampling.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files are present")
    return True


def validate_yaml_config():
    """Validate YAML configuration syntax and structure."""
    print("Validating YAML configuration...")
    
    config_path = Path(__file__).parent / "config" / "gspo_dynamic_sampling.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key sections
        required_sections = ['algorithm', 'actor_rollout_ref', 'data', 'trainer']
        for section in required_sections:
            if section not in config:
                print(f"  ✗ Missing section: {section}")
                return False
            print(f"  ✓ Section present: {section}")
        
        # Check dynamic sampling config
        if 'filter_groups' not in config['algorithm']:
            print("  ✗ Missing filter_groups in algorithm section")
            return False
        
        filter_groups = config['algorithm']['filter_groups']
        if not filter_groups.get('enable', False):
            print("  ✗ Dynamic sampling not enabled")
            return False
        
        print(f"  ✓ Dynamic sampling enabled with metric: {filter_groups.get('metric', 'N/A')}")
        
        # Check GSPO config
        actor_config = config['actor_rollout_ref']['actor']
        if actor_config.get('policy_loss', {}).get('loss_mode') != 'gspo':
            print("  ✗ Loss mode is not set to 'gspo'")
            return False
        
        print(f"  ✓ GSPO loss mode configured")
        
        clip_low = actor_config.get('clip_ratio_low', 0)
        clip_high = actor_config.get('clip_ratio_high', 0)
        if clip_low >= clip_high:
            print(f"  ✗ Invalid clipping ratios: low={clip_low}, high={clip_high}")
            return False
        
        print(f"  ✓ Valid clipping ratios: low={clip_low}, high={clip_high}")
        
        print("✓ YAML configuration is valid")
        return True
        
    except yaml.YAMLError as e:
        print(f"  ✗ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Configuration validation error: {e}")
        return False


def validate_shell_script():
    """Validate shell script syntax and key configurations."""
    print("Validating shell script...")
    
    script_path = Path(__file__).parent / "gspo_dynamic_sampling_3b_math.sh"
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for key GSPO parameters
        gspo_params = [
            'loss_mode=gspo',
            'enable_filter_groups=true',
            'algorithm.filter_groups.enable=',
            'algorithm.filter_groups.metric=',
            'clip_ratio_low=',
            'clip_ratio_high='
        ]
        
        missing_params = []
        for param in gspo_params:
            if param not in content:
                missing_params.append(param)
            else:
                print(f"  ✓ Parameter found: {param}")
        
        if missing_params:
            print(f"  ✗ Missing parameters: {missing_params}")
            return False
        
        # Check for proper script structure
        if not content.startswith('#!/usr/bin/env bash'):
            print("  ✗ Missing proper shebang")
            return False
        
        if 'set -xeuo pipefail' not in content:
            print("  ✗ Missing error handling")
            return False
        
        print("✓ Shell script is properly configured")
        return True
        
    except Exception as e:
        print(f"  ✗ Shell script validation error: {e}")
        return False


def validate_python_script():
    """Validate Python script syntax."""
    print("Validating Python script...")
    
    script_path = Path(__file__).parent / "main_gspo_dynamic.py"
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Basic syntax check
        try:
            compile(content, script_path, 'exec')
            print("  ✓ Python syntax is valid")
        except SyntaxError as e:
            print(f"  ✗ Python syntax error: {e}")
            return False
        
        # Check for key imports and functions
        key_elements = [
            '@hydra.main',
            'def main(',
            'def run_gspo_training(',
            'RayPPOTrainer',
            'Dynamic Sampling'
        ]
        
        missing_elements = []
        for element in key_elements:
            if element not in content:
                missing_elements.append(element)
            else:
                print(f"  ✓ Element found: {element}")
        
        if missing_elements:
            print(f"  ✗ Missing elements: {missing_elements}")
            return False
        
        print("✓ Python script is properly structured")
        return True
        
    except Exception as e:
        print(f"  ✗ Python script validation error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("GSPO Dynamic Sampling Implementation Validation")
    print("=" * 80)
    
    validators = [
        validate_file_structure,
        validate_yaml_config,
        validate_shell_script,
        validate_python_script,
    ]
    
    results = []
    for validator in validators:
        print(f"\\n{'-' * 60}")
        result = validator()
        results.append(result)
        print(f"{'-' * 60}")
    
    print(f"\\n{'=' * 80}")
    print("Validation Results Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (validator, result) in enumerate(zip(validators, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {validator.__name__}: {status}")
    
    print(f"\\nTotal: {passed}/{total} validations passed")
    
    if passed == total:
        print("\\n🎉 Implementation validation successful!")
        print("\\nNext steps:")
        print("1. Install verl dependencies: pip install -r requirements.txt")  
        print("2. Run the training script: bash recipe/gspo/gspo_dynamic_sampling_3b_math.sh")
        print("3. Or use Python script: python recipe/gspo/main_gspo_dynamic.py")
        return 0
    else:
        print(f"\\n❌ {total - passed} validation(s) failed. Please fix the issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)