"""
Key Components:
1. Data Loading and Processing
2. GPT-3/4 API Interaction
3. Prompt Construction and Management
4. Result Evaluation and Storage

v2:
test with basic pipeline. just retriever and answer.

renamed to main.py

"""

import os
import json
import argparse
import random
import time

# Standard ML libraries
import numpy as np
import torch
import torch.nn.functional as F
import openai
import os
import copy

# Import custom modules



if __name__ == '__main__':
    """
    Main execution block.
    
    The workflow is:
    1. Parse arguments and set up environment
    2. Initialize RL algorithm
    3. Load and process data
    4. Run inference with RL-based retrieval
    5. Save results
    """
    # Parse arguments
    from run_gpt3_comment import parse_args

    args_list = [
        '--data_root_train', './data/nq_tmp/train.json',
        '--data_root_test', './data/nq_tmp/test.json',
        '--output_root', './data/nq_tmp/results',
    ]

    # Parse arguments
    args = parse_args(args_list)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    
