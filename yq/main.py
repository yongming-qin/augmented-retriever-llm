"""
Improve hallucination using RL.
load data
run RL algorithm
save results
2025-05-26
Yongming Qin, Chenxi Chen

Key Components:
1. Data Loading and Processing
2. GPT-3/4 API Interaction
3. Prompt Construction and Management
4. Result Evaluation and Storage


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

# Import custom modules


"""
Main execution block.

The workflow is:
1. Load and process data
2. Initialize RL algorithm
4. Run inference with RL-based retrieval
5. Save results
"""


"""
three multi-choice datasets:
    biomedical dataset: Pubmedqa
    speech detection dataset: ethos-national
    climate change dataset: eval-climate
two open-ended QA dataset:
    T=REx
    natural quesitons NatQA
    
KILT dataset:

    
"""

def load_data(path):
    """
    Load and process the KILT Natural Questions dataset.
    Returns a list of dictionaries containing question-answer pairs with provenance information.
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)
    
    # Print some sample data
    print(f"\nLoaded {len(data)} samples from KILT Natural Questions dataset")
    print("\nSample 1:")
    print(f"Question: {data[0]['input']}")
    print(f"Answer: {data[0]['output'][0]['answer']}")
    print(f"Provenance: {data[0]['output'][0]['provenance'][0]['title']}")
    
    print("\nSample 2:")
    print(f"Question: {data[1]['input']}")
    print(f"Answer: {data[1]['output'][0]['answer']}")
    print(f"Provenance: {data[1]['output'][0]['provenance'][0]['title']}")
    
    return data


if __name__ == '__main__':
    load_data('/home/yq/ssd/hallucination/kilt-meta/nq-train-kilt.jsonl')


