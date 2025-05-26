"""
This file implements the main inference pipeline for the RL-augmented retriever system.
It combines a reinforcement learning-based retriever with GPT-3/4 to improve few-shot learning performance.

Key Components:
1. Data Loading and Processing
2. RL-based Retriever Integration
3. GPT-3/4 API Interaction
4. Prompt Construction and Management
5. Result Evaluation and Storage

The workflow is:
1. Load and process training/test data
2. Initialize the RL-based retriever
3. For each test question:
   - Use RL retriever to select most relevant examples
   - Construct prompt with selected examples
   - Send to GPT-3/4 for answer generation
   - Evaluate and store results
"""

import os
import json
import argparse
import random
import time

# Import custom modules
from base_prompt import *  # Contains prompt templates and formatting functions
from model import *  # Contains the policy network model definition
from utilities import extract_prediction, normalize_answer, _strip_string, read_json_all, last_boxed_only_string
from algorithm import init_algorithm  # Initializes different RL algorithms
from retriever.bm25_retriever import bm25_retrieve  # BM25-based retrieval as baseline
from utils.clusterfunc import step1, step2, step3  # Clustering utilities
from utils.hints import hint_aug, seed_aug  # Data augmentation utilities

# Standard ML libraries
import numpy as np
import torch
import torch.nn.functional as F
import openai
import os
import copy

# Azure OpenAI specific imports
import httpx
from openai import AzureOpenAI
from retriever.bm25_retriever import bm25_retrieve

# Azure authentication setup
os.environ["APP_CLIENT_ID"] = "long-tail-knowledge-app"
os.environ["APP_CLIENT_SECRET"] = "zyAeZTrIEJli3cqyVD2jvTDia6Ua"

from llm_idam_token_generator.idam_token_generator import get_llm_access_token

# Azure OpenAI API configuration
OPENAI_ENDPOINT = "https://openai-llm-frontdoor-hma7evbthrd4cugn.a01.azurefd.net"
OPENAI_DEPLOYMENT_MODEL = "gpt-4-32k-beta"
OPENAI_AZURE_API_VERSION = "2023-12-01-preview"
APIM_KEY = "8b96051ed6b84e4dad762fdc9f8c809e"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="xxx",  # Placeholder, actual key not used
    azure_endpoint=OPENAI_ENDPOINT,
    azure_deployment=OPENAI_DEPLOYMENT_MODEL,
    api_version=OPENAI_AZURE_API_VERSION,
    http_client=httpx.Client(verify=False),
    default_headers={
        'Authorization': f'Bearer {get_llm_access_token()}',
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
    }
)

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
            
    elif 'MATH' in args.data_root_test:
        # Load MATH problems
        problems_test = read_json_all(args.data_root_test)
        if 'pth' in args.data_root_train:
            problems_train = torch.load(args.data_root_train)
        else:
            problems_train = read_json_all(args.data_root_train)
        problems = problems_test + problems_train
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
        
    elif args.data_root_test in ['ethos-national_origin', 'blimp-anaphor_number_agreement', 'tweet_eval-irony', 'wino_grande', 'race-middle', 'proto_qa', 'race-high', 'kilt_hotpotqa', 'ade_corpus_v2-classification', 'hate_speech18', 'tweet_eval-stance_hillary', 'tweet_eval-offensive', 'kilt_ay2', 'squad-with_context', 'tweet_eval-sentiment', 'ethos-religion', 'wikisql', 'squad-no_context', 'ethos-race', 'tweet_eval-stance_climate', 'hate_speech_offensive', 'kilt_nq', 'tweet_eval-hate', 'tweet_eval-emotion', 'ethos-sexual_orientation', 'hatexplain', 'kilt_fever', 'blimp-ellipsis_n_bar_2', 'ethos-gender', 'ade_corpus_v2-dosage', 'ethos-disability', 'tweet_eval-stance_atheism', 'kilt_zsre', 'blimp-sentential_negation_npi_licensor_present', 'tweet_eval-stance_feminist', 'kilt_trex', 'tweet_eval-stance_abortion', 'ethos-directed_vs_generalized', 'blimp-sentential_negation_npi_scope', 'tweet_eval-emoji']:
        # Load various NLP tasks
        with open('../data/metaicl/task_data_splits.json', 'r', encoding='utf-8') as json_file:
            train_test_split = json.load(json_file)
        problems_test = train_test_split[args.data_root_test]['test']
        problems_train = train_test_split[args.data_root_test]['train']
        problems = problems_test + problems_train
        test_pids = list(i for i in range(len(problems_test)))
        train_pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
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
            {
                "role": "user", "content": user_prompt
            }],
            temperature=0.0,
            max_tokens=args.max_tokens,
            top_p=1,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty
        )
        output = response.choices[0].message.content
        
        # Process output
        if output is not None:
            if output.startswith("\n\n"):
                output = output[2:]
            output = output.split("\n")[0]
    else:
        # Handle other GPT models
        response = openai.Completion.create(
            engine=args.engine,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            stop=["\n"]
        )
        output = response["choices"][0]["text"].strip()

    return output

def get_result_file(args):
    """
    Generate the path for saving results.
    
    Args:
        args: Command line arguments containing output settings
        
    Returns:
        result_file: Path to the result file
    """
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(
        result_path,
        args.label,
        args.test_split,
        args.prompt_format,
        args.shot_number,
        args.seed
    )

    return result_file

def save_results(result_file, acc, correct, count, cand_pids, args, results):
    """
    Save experiment results to a JSON file.
    
    Args:
        result_file: Path to save the results
        acc: Accuracy of the model
        correct: Number of correct predictions
        count: Total number of predictions
        cand_pids: List of candidate problem IDs
        args: Command line arguments
        results: Detailed results for each test case
    """
    data = {
        'acc': acc,
        'correct': correct,
        'count': count,
        'cand_pids': cand_pids,
        'args': vars(args),
        'results': results
    }

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

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
        
        # Handle false test IDs if requested
        if args.get_false_test_pids:
            false_test_pids = []
            correct_test_pids = []
            if args.test_pids_ckpt:
                result_key = pids
            else:
                result_key = results.keys()
            for pid in result_key:
                if results[pid]["true_false"] == False:
                    false_test_pids.append(pid)
                else:
                    correct_test_pids.append(pid)
            print("false test num: {}".format(len(false_test_pids)))
            print("accuracy for test_pids: {}".format(len(correct_test_pids) / len(result_key)))
            exit(0)
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

    # Handle training checkpoints
    if args.train_ckpt:
        train_pids = torch.load(args.train_ckpt)
        train_pids = [i for i in train_pids if i not in pids]
        val_examples = []
        for pid in train_pids:
            val_example = create_example_from_pid(pid, problems, args, test=True)
            val_examples.append(val_example)
            
        if args.val_ckpt:
            val_pids = torch.load(args.val_ckpt)
            correct_val_pids = [i for i in val_pids if i not in train_pids]
            correct_val_examples = []
            for pid in correct_val_pids:
                correct_val_example = create_example_from_pid(pid, problems, args, test=True)
                correct_val_examples.append(correct_val_example)

    # Handle data augmentation
    if args.aug_method == 'hints':
        # Implement hint-based augmentation
        aug_label = []
        for i in cluster_num.keys():
            if cluster_num[i] <= args.aug_th:
                aug_label.append(i)
        print("aug label", aug_label)
        cluster_train_id = [i for i in range(len(cluster_id)) if cluster_id[i] in aug_label]
        
        examples_train_aug = [cand_examples[i] for i in cluster_train_id]
        embeddings = [embeddings[i] for i in cluster_train_id]
        cluster_id_aug = [cluster_id[i] for i in cluster_train_id]
        
        print("len train", len(examples_train_aug))
        print("len embeddings", len(embeddings))
        print("len cluster id", len(cluster_id_aug))
        new_samples = hint_aug(args, examples_train_aug, embeddings, cluster_id_aug)
        
    elif args.aug_method == 'seed':
        # Implement seed-based augmentation
        seed_sentence = torch.load(f"cluster_results/seed_sentence_{args.cluster_type}_{args.n_clusters}_1000.pt")
        print(f"Load seed sentence from cluster_results/seed_sentence_{args.cluster_type}_{args.n_clusters}_1000.pt")
        print("generate augmented samples using seed sentences.")
        new_samples = seed_aug(args, seed_sentence)
    else:
        new_samples = []

    # Process augmented samples
    new_pids = []
    for i in range(len(new_samples)):
        if i not in cand_pids:
            new_pids.append(i)
        else:
            print("id already exists!")
            exit(0)

    # Combine original and augmented samples
    cand_examples.extend(new_samples)
    cand_pids.extend(new_pids)
    for i in range(len(new_samples)):
        problems.update({new_pids[i]: {'new_samples': new_samples[i]}})
    print("Extended cand size", len(cand_pids))

    # Prepare test examples
    test_examples = []
    for pid in pids:
        if 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin', ...]:
            pid = int(pid)
        example = create_example_from_pid(pid, problems, args, test=True)
        test_examples.append(example)

    # Load model checkpoints if provided
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        if args.ckpt_context:
            ckpt_context_path = os.path.join(args.ckpt_root, args.ckpt_context)
            if os.path.exists(ckpt_context_path):
                algorithm.context_net.linear.load_state_dict(torch.load(ckpt_context_path))
                print("context model loaded")
            else:
                print(f"The ckpt path for [{ckpt_context_path}] does not exist!")
        if args.ckpt_lossnet:
            ckpt_lossnet_path = os.path.join(args.ckpt_root, args.ckpt_lossnet)
            if os.path.exists(ckpt_lossnet_path):
                algorithm.learned_loss_net.load_state_dict(torch.load(ckpt_lossnet_path))
                print("Loss net loaded")
            else:
                print(f"The ckpt path for [{ckpt_lossnet_path}] does not exist!")

        if os.path.exists(ckpt_path):
            algorithm.model.linear.load_state_dict(torch.load(ckpt_path))
            print("Policy model loaded")
        else:
            print(f"The ckpt path for [{ckpt_path}] does not exist!")

    else:
        print(f"!!! Load the pre-traind model instead!")

    # Set model to evaluation mode
    algorithm.model.eval()

    # Calculate embeddings for candidate examples
    cand_embedding = algorithm.predict(cand_examples)
    if args.train_ckpt:
        val_embedding = algorithm.predict(val_examples)
        if args.val_ckpt:
            correct_val_embedding = algorithm.predict(correct_val_examples)

    # Initialize tracking variables
    wrong_max_scores = []
    correct_max_scores = []
    wrong_max_scores_true = []
    correct_max_scores_true = []
    shot_len_avg = []

    # Store original candidate IDs if preselection is enabled
    if args.preselection:
        original_cand_pids = copy.deepcopy(cand_pids)

    # Main inference loop
    with torch.no_grad():
        for i, pid in enumerate(pids):
            # Convert pid to int for certain datasets
            if 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin', ...]:
                pid = int(pid)
            
            count = i + 1  # number of current results
            problem = problems[pid]

            # Extract answer based on dataset type
            if 'tabmwp' in args.data_root_test or 'medqa' in args.data_root_test:
                answer = problem['answer']
            elif 'gsm8k' in args.data_root_test:
                _, _, answer = problem['answer'].partition("\n#### ")
            elif 'MATH' in args.data_root_test:
                answer = remove_boxed(last_boxed_only_string(problem["solution"]))
            elif 'pubmed' in args.data_root_test:
                answer = problem['final_decision']
            elif "output" in problem.keys():
                answer = problem['output']
            else:
                raise Exception("The dataset does not exist!")

            # Extract options if available
            if "options" in problem.keys():
                if 'medqa' in args.data_root_test:
                    options = []
                    for o in problems[pid]['options'].keys():
                        options.append(problems[pid]['options'][o])
                else:
                    options = problems[pid]['options']
            elif "choices" in problem.keys():
                options = problems[pid]['choices']
            else:
                options = None

            # Extract unit if available
            if 'unit' in problems[pid].keys():
                unit = problems[pid]['unit']
            else:
                unit = None

            # Check if result already exists
            if str(pid) in results:
                pid = str(pid)
                output = results[pid]["output"]
                shot_len_avg.append(len(results[pid]["shot_pids"]))
            else:
                # Create example for current problem
                example = create_example_from_pid(pid, problems, args, test=True)

                # Handle preselection if enabled
                if args.preselection:
                    Cand_example = bm25_retrieve(example, cand_examples, n=args.select_number)
                    cand_idx = bm25_retrieve(example, cand_examples, n=args.select_number, return_index=True).tolist()
                    cand_pids = [original_cand_pids[c] for c in cand_idx]
                    cand_embedding = algorithm.predict(Cand_example)

                # Get context embedding
                if args.gamma != 0:
                    ctxt_embedding, _ = algorithm.predict([example], test=True)
                else:
                    ctxt_embedding = algorithm.predict([example], test=True)

                # Calculate similarity scores
                scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0]  # [cand_num]
                scores = scores.cpu().detach().numpy().tolist()

                # Handle validation scores if training checkpoint is provided
                score_th = args.score_th
                if args.train_ckpt:
                    val_scores = F.softmax(torch.mm(val_embedding, cand_embedding.t()), dim=1)  # [cand_num]
                    val_scores = val_scores.cpu().detach().numpy()
                    val_mean, val_std = np.mean(val_scores, axis=0), np.std(val_scores, axis=0)
                    score_th = (args.score_th - val_mean) + scores

                # Handle correct validation scores if validation checkpoint is provided
                if args.val_ckpt:
                    correct_val_scores = F.softmax(torch.mm(ctxt_embedding, correct_val_embedding.t()), dim=1)[0]
                    correct_val_scores = correct_val_scores.cpu().detach().numpy().tolist()

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

                shot_len_avg.append(len(shot_pids))

                # Build prompt without test example
                prompt_no_test = build_prompt(problems, shot_pids, pid, args, include_test=False)

                # Build final prompt with test example
                prompt = build_prompt(problems, shot_pids, pid, args)

                # Get GPT output
                output = get_gpt_output(prompt, args)
                
        ########################################################
        # 
        ########################################################
                
                
            # Process output based on dataset type
            if 'tabmwp' in args.data_root_test or 'medqa' in args.data_root_test:
                # Extract prediction and normalize
                if output:
                    prediction = extract_prediction(output, options, args.option_inds)
                    prediction_norm = normalize_answer(prediction, unit)
                else:
                    prediction = output
                    prediction_norm = output

                # Normalize answer
                answer_norm = normalize_answer(answer, unit)

            elif 'pubmed' in args.data_root_test:
                answer_norm = answer
                prediction = extract_prediction(output, options, args.option_inds)
                prediction_norm = prediction
            else:
                # Handle other dataset types
                if output:
                    prediction = extract_prediction(output, options, args.option_inds)
                    prediction_norm = normalize_answer(prediction, unit)
                else:
                    prediction = output
                    prediction_norm = output

                # Normalize answer
                answer_norm = normalize_answer(answer, unit)

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

            # Update accuracy
            if prediction_norm == answer_norm:
                correct += 1

            # Print progress
            if count % args.save_every == 0:
                acc = correct / count
                print(f"pid: {pid}, count: {count}, correct: {correct}, acc: {acc}")
                save_results(result_file, acc, correct, count, cand_pids, args, results)

    # Save final results
    acc = correct / total
    print(f"pid: {pid}, count: {count}, correct: {correct}, acc: {acc}")
    save_results(result_file, acc, correct, count, cand_pids, args, results) 