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
ProRL Data Preparation Script

Prepares diverse task datasets for ProRL training as described in the paper:
- Math problems (GSM8K, MATH, etc.)
- Code problems (HumanEval, MBPP, etc.)
- STEM reasoning (GPQA, etc.)
- Logic puzzles (Reasoning Gym)
- Instruction following (IFEval, etc.)
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random


def create_math_dataset(output_path: str, num_samples: int = 10000):
    """Create math reasoning dataset."""
    print(f"Creating math dataset with {num_samples} samples...")
    
    # Example math problems (in practice, load from actual datasets)
    math_problems = []
    
    # GSM8K style problems
    for i in range(num_samples // 4):
        problem = {
            "prompt": f"Solve this math problem step by step:\n\nProblem {i+1}: A store has 120 apples. If they sell 45 apples in the morning and 38 apples in the afternoon, how many apples do they have left?",
            "response": "Let me solve this step by step.\n\nInitially, the store has 120 apples.\n\nIn the morning, they sell 45 apples.\nRemaining apples = 120 - 45 = 75 apples\n\nIn the afternoon, they sell 38 apples.\nFinal remaining apples = 75 - 38 = 37 apples\n\nTherefore, the store has 37 apples left.",
            "reward": 1.0,
            "task_type": "math",
            "difficulty": "easy"
        }
        math_problems.append(problem)
    
    # MATH style problems
    for i in range(num_samples // 4):
        problem = {
            "prompt": f"Solve this algebra problem:\n\nFind the value of x if 3x + 7 = 22.",
            "response": "I need to solve for x in the equation 3x + 7 = 22.\n\nStep 1: Subtract 7 from both sides\n3x + 7 - 7 = 22 - 7\n3x = 15\n\nStep 2: Divide both sides by 3\n3x ÷ 3 = 15 ÷ 3\nx = 5\n\nTherefore, x = 5.",
            "reward": 1.0,
            "task_type": "math",
            "difficulty": "medium"
        }
        math_problems.append(problem)
    
    # Add more complex problems
    for i in range(num_samples // 2):
        problem = {
            "prompt": f"Advanced problem {i+1}: Solve the system of equations:\n2x + 3y = 12\n4x - y = 5",
            "response": "I'll solve this system of equations using substitution method.\n\nFrom equation 2: 4x - y = 5\nSolving for y: y = 4x - 5\n\nSubstituting into equation 1:\n2x + 3(4x - 5) = 12\n2x + 12x - 15 = 12\n14x = 27\nx = 27/14\n\nNow finding y:\ny = 4(27/14) - 5 = 108/14 - 70/14 = 38/14 = 19/7\n\nTherefore: x = 27/14, y = 19/7",
            "reward": 1.0,
            "task_type": "math",
            "difficulty": "hard"
        }
        math_problems.append(problem)
    
    # Save dataset
    with open(output_path, 'w') as f:
        for problem in math_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"Math dataset saved to {output_path}")


def create_code_dataset(output_path: str, num_samples: int = 5000):
    """Create code generation dataset."""
    print(f"Creating code dataset with {num_samples} samples...")
    
    code_problems = []
    
    # Basic programming problems
    for i in range(num_samples // 3):
        problem = {
            "prompt": f"Write a Python function to {['reverse a string', 'find the maximum in a list', 'check if a number is prime'][i % 3]}.",
            "response": """def solve_problem():
    # Implementation here
    if problem_type == 'reverse':
        def reverse_string(s):
            return s[::-1]
        return reverse_string
    elif problem_type == 'maximum':
        def find_max(lst):
            return max(lst) if lst else None
        return find_max
    else:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        return is_prime""",
            "reward": 1.0,
            "task_type": "code",
            "difficulty": "easy"
        }
        code_problems.append(problem)
    
    # Algorithm problems
    for i in range(num_samples // 3):
        problem = {
            "prompt": "Implement a binary search algorithm to find a target value in a sorted array.",
            "response": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found""",
            "reward": 1.0,
            "task_type": "code",
            "difficulty": "medium"
        }
        code_problems.append(problem)
    
    # Complex problems
    for i in range(num_samples // 3):
        problem = {
            "prompt": "Implement a solution for the longest common subsequence problem using dynamic programming.",
            "response": """def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]""",
            "reward": 1.0,
            "task_type": "code",
            "difficulty": "hard"
        }
        code_problems.append(problem)
    
    with open(output_path, 'w') as f:
        for problem in code_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"Code dataset saved to {output_path}")


def create_stem_dataset(output_path: str, num_samples: int = 3000):
    """Create STEM reasoning dataset."""
    print(f"Creating STEM dataset with {num_samples} samples...")
    
    stem_problems = []
    
    for i in range(num_samples):
        domains = ["physics", "chemistry", "biology"]
        domain = domains[i % 3]
        
        if domain == "physics":
            problem = {
                "prompt": "A ball is thrown upward with an initial velocity of 20 m/s. How high will it go? (g = 9.8 m/s²)",
                "response": "Using kinematic equation: v² = u² + 2as\n\nAt maximum height, final velocity v = 0\nInitial velocity u = 20 m/s\nAcceleration a = -g = -9.8 m/s²\n\n0² = 20² + 2(-9.8)s\n0 = 400 - 19.6s\ns = 400/19.6 ≈ 20.4 m\n\nThe ball will reach a maximum height of approximately 20.4 meters.",
                "reward": 1.0,
                "task_type": "stem",
                "domain": "physics"
            }
        elif domain == "chemistry":
            problem = {
                "prompt": "Balance the chemical equation: C₄H₁₀ + O₂ → CO₂ + H₂O",
                "response": "To balance C₄H₁₀ + O₂ → CO₂ + H₂O:\n\n1. Balance carbon: C₄H₁₀ + O₂ → 4CO₂ + H₂O\n2. Balance hydrogen: C₄H₁₀ + O₂ → 4CO₂ + 5H₂O\n3. Balance oxygen: Count O atoms on right: 4(2) + 5(1) = 13\n   Need 13/2 = 6.5 O₂ molecules\n   Multiply by 2: 2C₄H₁₀ + 13O₂ → 8CO₂ + 10H₂O\n\nBalanced equation: 2C₄H₁₀ + 13O₂ → 8CO₂ + 10H₂O",
                "reward": 1.0,
                "task_type": "stem",
                "domain": "chemistry"
            }
        else:  # biology
            problem = {
                "prompt": "Explain the process of photosynthesis and write its overall equation.",
                "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.\n\nProcess:\n1. Light reactions: Chlorophyll absorbs light, splitting water molecules\n2. Calvin cycle: CO₂ is fixed into glucose using ATP and NADPH\n\nOverall equation:\n6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nThis process produces glucose (food) and oxygen as a byproduct.",
                "reward": 1.0,
                "task_type": "stem",
                "domain": "biology"
            }
        
        stem_problems.append(problem)
    
    with open(output_path, 'w') as f:
        for problem in stem_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"STEM dataset saved to {output_path}")


def create_logic_puzzles_dataset(output_path: str, num_samples: int = 2000):
    """Create logic puzzles dataset."""
    print(f"Creating logic puzzles dataset with {num_samples} samples...")
    
    logic_problems = []
    
    for i in range(num_samples):
        puzzle_types = ["sudoku", "logic_grid", "pattern"]
        puzzle_type = puzzle_types[i % 3]
        
        if puzzle_type == "sudoku":
            problem = {
                "prompt": "Solve this 4x4 Sudoku puzzle:\n[1, _, 3, _]\n[_, 2, _, 4]\n[3, _, 1, _]\n[_, 4, _, 2]\nFill in the blanks with numbers 1-4.",
                "response": "Analyzing the 4x4 Sudoku:\n\nRow 1: [1, _, 3, _] - missing 2, 4\nRow 2: [_, 2, _, 4] - missing 1, 3\nRow 3: [3, _, 1, _] - missing 2, 4\nRow 4: [_, 4, _, 2] - missing 1, 3\n\nColumn analysis and elimination:\nSolution:\n[1, 2, 3, 4]\n[3, 2, 4, 4]\n[3, 1, 1, 2]\n[1, 4, 3, 2]",
                "reward": 1.0,
                "task_type": "logic",
                "puzzle_type": "sudoku"
            }
        elif puzzle_type == "logic_grid":
            problem = {
                "prompt": "Three friends (Alice, Bob, Carol) have different pets (cat, dog, fish) and live in different cities (New York, Boston, Chicago). Alice doesn't live in New York. The person with the cat lives in Boston. Bob has a dog. Who lives where and has what pet?",
                "response": "Let me solve this step by step:\n\nGiven:\n- Alice doesn't live in New York\n- Person with cat lives in Boston\n- Bob has a dog\n\nDeduction:\n1. Bob has a dog (given)\n2. Since Bob has a dog, he doesn't have the cat\n3. The cat owner lives in Boston\n4. Since Bob doesn't have the cat, Bob doesn't live in Boston\n5. Alice doesn't live in New York (given)\n6. So Alice lives in Boston or Chicago\n7. Since cat owner lives in Boston, if Alice lives in Boston, she has the cat\n\nSolution:\n- Alice: Cat, Boston\n- Bob: Dog, Chicago (or New York)\n- Carol: Fish, remaining city",
                "reward": 1.0,
                "task_type": "logic",
                "puzzle_type": "logic_grid"
            }
        else:  # pattern
            problem = {
                "prompt": "What comes next in this sequence: 2, 6, 12, 20, 30, ?",
                "response": "Analyzing the pattern:\n2, 6, 12, 20, 30, ?\n\nDifferences:\n6-2 = 4\n12-6 = 6\n20-12 = 8\n30-20 = 10\n\nThe differences are: 4, 6, 8, 10\nThese increase by 2 each time.\n\nNext difference: 10 + 2 = 12\nNext number: 30 + 12 = 42\n\nAlternatively, this follows n(n+1) where n = 1,2,3,4,5,6:\n1×2=2, 2×3=6, 3×4=12, 4×5=20, 5×6=30, 6×7=42",
                "reward": 1.0,
                "task_type": "logic",
                "puzzle_type": "pattern"
            }
        
        logic_problems.append(problem)
    
    with open(output_path, 'w') as f:
        for problem in logic_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"Logic puzzles dataset saved to {output_path}")


def create_instruction_following_dataset(output_path: str, num_samples: int = 3000):
    """Create instruction following dataset."""
    print(f"Creating instruction following dataset with {num_samples} samples...")
    
    instruction_problems = []
    
    for i in range(num_samples):
        instruction_types = ["format", "length", "style", "constraint"]
        instr_type = instruction_types[i % 4]
        
        if instr_type == "format":
            problem = {
                "prompt": "Write a brief summary of renewable energy. Format your response as exactly 3 bullet points.",
                "response": "• Solar energy harnesses sunlight through photovoltaic panels to generate clean electricity\n• Wind power uses turbines to convert wind movement into sustainable electrical energy\n• Hydroelectric systems utilize flowing water to produce renewable electricity through dams",
                "reward": 1.0,
                "task_type": "instruction_following",
                "constraint_type": "format"
            }
        elif instr_type == "length":
            problem = {
                "prompt": "Explain photosynthesis in exactly 50 words.",
                "response": "Photosynthesis is the biological process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen. Chlorophyll in plant leaves captures light energy, driving chemical reactions that produce food for the plant while releasing oxygen as a byproduct, essential for life on Earth.",
                "reward": 1.0,
                "task_type": "instruction_following",
                "constraint_type": "length"
            }
        elif instr_type == "style":
            problem = {
                "prompt": "Explain gravity in the style of a pirate captain.",
                "response": "Ahoy matey! Listen well to this tale of gravity, the invisible force that keeps ye boots planted on me ship's deck! 'Tis the same mysterious power that pulls yer treasure chest to the ocean floor and keeps the moon dancin' around our fair Earth. Sir Newton discovered this mighty force, and by me beard, it affects every scurvy sailor and landlubber alike!",
                "reward": 1.0,
                "task_type": "instruction_following",
                "constraint_type": "style"
            }
        else:  # constraint
            problem = {
                "prompt": "Write a paragraph about dogs without using the letter 'a'.",
                "response": "Dogs bring joy to people everywhere. These furry friends show loyalty beyond belief. Dogs come in different sizes, from tiny terriers to huge hounds. They help humans in countless methods, working with police or guiding blind people. Dogs need proper food, exercise, love from owners to live good lives.",
                "reward": 1.0,
                "task_type": "instruction_following",
                "constraint_type": "no_letter_a"
            }
        
        instruction_problems.append(problem)
    
    with open(output_path, 'w') as f:
        for problem in instruction_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"Instruction following dataset saved to {output_path}")


def create_validation_dataset(output_path: str, num_samples: int = 1000):
    """Create validation dataset with mixed tasks."""
    print(f"Creating validation dataset with {num_samples} samples...")
    
    val_problems = []
    task_types = ["math", "code", "stem", "logic", "instruction"]
    
    for i in range(num_samples):
        task_type = task_types[i % 5]
        
        # Simple validation examples for each task type
        if task_type == "math":
            problem = {
                "prompt": f"Solve: {random.randint(10, 50)} + {random.randint(10, 50)} = ?",
                "response": "Let me add these numbers step by step...",
                "reward": 1.0,
                "task_type": "math",
                "split": "validation"
            }
        elif task_type == "code":
            problem = {
                "prompt": "Write a function to add two numbers.",
                "response": "def add_numbers(a, b):\n    return a + b",
                "reward": 1.0,
                "task_type": "code",
                "split": "validation"
            }
        elif task_type == "stem":
            problem = {
                "prompt": "What is the chemical symbol for water?",
                "response": "The chemical symbol for water is H₂O, representing two hydrogen atoms bonded to one oxygen atom.",
                "reward": 1.0,
                "task_type": "stem",
                "split": "validation"
            }
        elif task_type == "logic":
            problem = {
                "prompt": "If all roses are flowers and all flowers are plants, are all roses plants?",
                "response": "Yes, all roses are plants. This follows from logical transitivity: roses → flowers → plants, therefore roses → plants.",
                "reward": 1.0,
                "task_type": "logic",
                "split": "validation"
            }
        else:  # instruction
            problem = {
                "prompt": "List three benefits of exercise.",
                "response": "Three benefits of exercise are: 1) Improved cardiovascular health, 2) Increased muscle strength and endurance, 3) Better mental health and mood regulation.",
                "reward": 1.0,
                "task_type": "instruction_following",
                "split": "validation"
            }
        
        val_problems.append(problem)
    
    with open(output_path, 'w') as f:
        for problem in val_problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"Validation dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ProRL training datasets")
    parser.add_argument("--output_dir", type=str, default="data/prorl_diverse_tasks", 
                       help="Output directory for datasets")
    parser.add_argument("--math_samples", type=int, default=10000, help="Number of math samples")
    parser.add_argument("--code_samples", type=int, default=5000, help="Number of code samples")
    parser.add_argument("--stem_samples", type=int, default=3000, help="Number of STEM samples")
    parser.add_argument("--logic_samples", type=int, default=2000, help="Number of logic samples")
    parser.add_argument("--instruction_samples", type=int, default=3000, help="Number of instruction samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Preparing ProRL diverse task datasets...")
    print(f"Output directory: {args.output_dir}")
    
    # Create datasets
    create_math_dataset(f"{args.output_dir}/math_problems.jsonl", args.math_samples)
    create_code_dataset(f"{args.output_dir}/code_problems.jsonl", args.code_samples)
    create_stem_dataset(f"{args.output_dir}/stem_problems.jsonl", args.stem_samples)
    create_logic_puzzles_dataset(f"{args.output_dir}/logic_puzzles.jsonl", args.logic_samples)
    create_instruction_following_dataset(f"{args.output_dir}/instruction_following.jsonl", args.instruction_samples)
    create_validation_dataset(f"{args.output_dir}/validation_data.jsonl", args.val_samples)
    
    print("\nDataset preparation completed!")
    print(f"Total training samples: {args.math_samples + args.code_samples + args.stem_samples + args.logic_samples + args.instruction_samples}")
    print(f"Validation samples: {args.val_samples}")
    print("\nDataset files created:")
    for filename in os.listdir(args.output_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(args.output_dir, filename)
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"  {filename}: {size:.2f} MB")


if __name__ == "__main__":
    main()