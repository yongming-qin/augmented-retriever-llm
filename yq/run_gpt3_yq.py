"""
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
import copy

# Import custom modules

from algorithm import init_algorithm # Initializes different RL algorithms


def load_data(args):
    """
    Load and process data from different sources and formats.
    
    Args:
        args: Command line arguments containing data paths and processing options
    
    Returns:
        problems: Dictionary of all problems (test + train)
        test_pids: List of test problem IDs
        cand_pids: List of candidate problem IDs for retrieval
    
    The function handles different data formats:
    1. MedQA format 
    2. MATH problems
    3. Various NLP tasks (squad, tweet_eval, etc.)
    4. Generic JSON format
    """
    if 'medqa' in args.data_root_test:
        # Load MedQA format data
        problems_test = [json.loads(line) for line in open(args.data_root_test, 'r')]
        problems_train = [json.loads(line) for line in open(args.data_root_train, 'r')]
        problems = problems_test + problems_train
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
        
        # Add validation data if provided
        if args.data_root_vali is not None:
            problems_vali = [json.loads(line) for line in open(args.data_root_vali, 'r')]
            problems += problems_vali
            
    else:
        # Load generic JSON format
        problems_test = json.load(open(args.data_root_test))
        problems_train = json.load(open(args.data_root_train))
        problems = {**problems_test, **problems_train}
        test_pids = list(problems_test.keys())
        train_pids = list(problems_train.keys())
        
    # Sample test problems if specified
    if args.test_number < len(test_pids) and args.test_number > 0:
        test_pids = random.sample(test_pids, args.test_number)
        
    # Load test IDs from checkpoint if provided
    if args.test_pids_ckpt:
        test_pids = torch.load(args.test_pids_ckpt)
    print(f"number of test problems: {len(test_pids)}\n")

    # Process candidate examples
    print(f"original cand set number {len(train_pids)}")
    if args.cand_ckpt:
        # Load candidate IDs from checkpoint
        cand_pids = torch.load(args.cand_ckpt)
        if 'MATH' in args.data_root_test:
            cand_pids = [i + len(problems_test) for i in cand_pids]
    else:
        # Sample candidate examples
        if args.cand_number < len(train_pids):
            cand_pids = random.sample(train_pids, args.cand_number)
        else:
            cand_pids = train_pids
            
    # Remove test examples from candidates
    cand_pids = [i for i in cand_pids if i not in test_pids]

    return problems, test_pids, cand_pids

def get_gpt_output(prompt, args):
    """
    Get response from GPT-3/4 model.
    
    Args:
        prompt: The input prompt for the model
        args: Command line arguments containing model settings
        
    Returns:
        output: The model's response text
        
    The function handles:
    1. Different model types (GPT-4, text-davinci-002, ada)
    2. Different prompt formats for different tasks
    3. Response processing and formatting
    """
    if "gpt4" in args.engine:
        # Handle GPT-4 specific settings
        if 'pubmed' in args.data_root_test:
            user_prompt = "Please answer yes or no or maybe for the question."
        elif 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin', ...]:
            user_prompt = "Please choose from all the options follow the given example."
        else:
            user_prompt = "Follow the given examples and answer the question following the same format."
            
        # Make API call to GPT-4
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_MODEL,
            messages=[{
                "role": "system", "content": prompt
            },
            
            
    

    
def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed command line arguments
        
    The arguments include:
    1. Data paths and settings
    2. Model configuration
    3. RL algorithm settings
    4. GPT-3/4 settings
    5. Training and evaluation parameters
    """
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--data_root_train', type=str, default='../data/tabmwp/problems_train.json')
    parser.add_argument('--data_root_test', type=str, default='../data/tabmwp/problems_test.json')
    parser.add_argument('--data_root_vali', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='../results')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list, 
                        default=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                                 "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
    parser.add_argument('--batch_size', type=int, default=50)
    
    # User options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--test_number', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format', type=str, default='Q-A')
    parser.add_argument('--shot_number', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='gpt4')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    
    # Policy Model settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased')
    parser.add_argument('--cand_number', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_context', type=str, default=None)
    parser.add_argument('--ckpt_lossnet', type=str, default=None)
    parser.add_argument('--cand_ckpt', type=str, default=None)
    parser.add_argument('--test_pids_ckpt', type=str, default=None)
    parser.add_argument('--train_ckpt', type=str, default=None)
    parser.add_argument('--val_ckpt', type=str, default=None)
    
    # RL Algorithm settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--adapt_bn', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--score_th', type=float, default=-1)
    
    # Data augmentation settings
    parser.add_argument('--cluster_type', type=str, default=None)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--aug_method', type=str, default=None)
    parser.add_argument('--preselection', action='store_true')
    parser.add_argument('--select_number', type=int, default=50)
    parser.add_argument('--get_false_test_pids', action='store_true')
    
    args = parser.parse_args()
    args.meta_batch_size = args.batch_size
    return args
    
    
    
    
    

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
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Initialize RL algorithm
    algorithm = init_algorithm(args)
    
    # Load data
    problems, pids, cand_pids = load_data(args)
    
    # Get result file path
    result_file = get_result_file(args)
    
    # Check for existing results
    if os.path.exists(result_file):
        print("# The result file exists! We will load the learned check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
        
    else:
        results = {}

    # Initialize counters
    total = len(pids)
    check_count = len(results)
    correct = 0

    # Prepare candidate examples
    print("candidate prompts: ")
    print("===========")
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)

    
    # Prepare test examples
    test_examples = []
    for pid in pids:
        if 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin', ...]:
            pid = int(pid)
        example = create_example_from_pid(pid, problems, args, test=True)
        test_examples.append(example)
        
    # Calculate embeddings for candidate examples
    cand_embedding = algorithm.predict(cand_examples)
    
    # Main inference loop
    with torch.no_grad():
        for i, pid in enumerate(pids):
            count = i + 1 # number of current results
            problem = problems[pid]

            answer = problem['answer']
            
            example = create_example_from_pid(pid, problems, args, test=True)
            
            ctxt_embedding = algorithm.predict([example], test=True)

            # Calculate similarity scores
            scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0] # [cand_num]
            scores = scores.cpu().detach().numpy().tolist()

            
            
            # Select examples based on scores
            shot_pids = []
            cand_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.shot_number]
            for cid in cand_ids[::-1]:
                if args.train_ckpt is None:
                    if scores[cid] > score_th:
                        shot_pids.append(cand_pids[cid])
                else:
                    if scores[cid] > score_th[cid]:
                        shot_pids.append(cand_pids[cid])
                        
                        
            # Build final prompt with test example
                prompt = build_prompt(problems, shot_pids, pid, args)

                # Get GPT output
                output = get_gpt_output(prompt, args)
                

            # Store results
            results[str(pid)] = {
                "output": output,
                "prediction": prediction,
                "prediction_norm": prediction_norm,
                "answer": answer,
                "answer_norm": answer_norm,
                "true_false": prediction_norm == answer_norm,
                "shot_pids": shot_pids
            }