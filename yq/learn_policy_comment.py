"""
Reinforcement Learning Policy for Few-shot Learning
This script implements a policy gradient method to learn optimal few-shot examples for question answering.
The policy network learns to select the most relevant examples to use as few-shot demonstrations.

Key Components:
1. Policy Network: A neural network that learns to select optimal few-shot examples
2. Reward Function: Based on whether the model's answer matches the ground truth
3. Policy Gradient Training: Uses REINFORCE algorithm to update the policy
4. GPT-3 Integration: Uses GPT-3 to generate answers based on selected few-shot examples

Author: Original implementation
Modified with detailed comments for learning purposes
"""

import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai

import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from tools import utils
from base_prompt import *
from model import *
from utilities import extract_prediction, normalize_answer

# Add parent directory to path for imports
sys.path.append("../")
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_data(args):
    """
    Load and prepare training data for the policy network.
    
    Args:
        args: Command line arguments containing data paths and parameters
        
    Returns:
        problems: Dictionary containing all problem data
        cand_pids: List of problem IDs to use as candidate few-shot examples
        train_pids: List of problem IDs to use for training the policy
    """
    # Load problem data from JSON file
    problems = json.load(open(os.path.join(args.data_root, f'problems_train.json')))
    problems = json.load(open(os.path.join(args.data_root, f'problems_train.json')))

    # Get list of all problem IDs
    pids = list(problems.keys())

    # Randomly sample problems for training and candidate examples
    samples = random.sample(pids, args.train_number + args.cand_number)
    train_pids = samples[:args.train_number]  # Problems to train policy on
    cand_pids = samples[args.train_number:]   # Problems to use as candidate few-shot examples

    return problems, cand_pids, train_pids


def get_gpt3_output(prompt, args):
    """
    Get output from GPT-3 model with specified parameters.
    
    Args:
        prompt: Input prompt for GPT-3
        args: Arguments containing GPT-3 parameters
        
    Returns:
        GPT-3's response text
    """
    return call_gpt3(args.engine, prompt, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                     args.presence_penalty)


@lru_cache(maxsize=10000)
def call_gpt3(engine, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    """
    Call GPT-3 API with caching to avoid redundant calls.
    Uses LRU cache to store recent API calls for efficiency.
    
    Args:
        engine: GPT-3 engine to use
        prompt: Input prompt
        temperature: Controls randomness (0 = deterministic)
        max_tokens: Maximum length of response
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for repeated tokens
        presence_penalty: Penalty for new tokens
        
    Returns:
        GPT-3's response text
    """
    patience = 100
    while True:
        try:
            # Make API call to GPT-3
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=["\n"])
            output = response["choices"][0]["text"].strip()
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! running out of patience waiting for OpenAI")
            else:
                time.sleep(0.1)
    return output


def get_batch_reward_loss(scores, cand_pids, pid_batch, option_batch, unit_batch, label_batch, args):
    """
    Calculate reward and loss for a batch of training examples.
    This is the core of the policy gradient algorithm.
    
    Args:
        scores: Policy network output scores for candidate examples
        cand_pids: List of candidate problem IDs
        pid_batch: Batch of current problem IDs
        option_batch: Batch of answer options
        unit_batch: Batch of units for answers
        label_batch: Batch of ground truth labels
        args: Command line arguments
        
    Returns:
        cids: Selected candidate IDs
        batch_reward: Total reward for the batch
        batch_loss: Policy gradient loss
    """
    batch_loss = 0
    batch_reward = 0

    # Process each example in the batch
    for i in range(len(scores)):
        # Convert scores to probabilities
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # Handle NaN values
        cand_prob /= cand_prob.sum()  # Normalize to get probabilities

        # Sample few-shot examples based on policy probabilities
        cids = np.random.choice(range(len(cand_pids)), args.shot_number, p=cand_prob, replace=False)
        cids = cids[::-1]  # Reverse to put more relevant examples closer to question
        shot_pids = [cand_pids[cid] for cid in cids]

        # Generate prompt with selected examples
        prompt = build_prompt(problems, shot_pids, pid_batch[i], args)

        # Get GPT-3's answer
        output = get_gpt3_output(prompt, args)
        prediction = extract_prediction(output, option_batch[i], args.option_inds)
        prediction_norm = normalize_answer(prediction, unit_batch[i])

        # Calculate log probability of selected examples
        log_prob = 0
        for cid in cids:
            log_prob += torch.log(scores[i, cid])

        # Calculate reward (1 for correct answer, -1 for incorrect)
        _reward = 1 if prediction_norm.lower() == label_batch[i].lower() else -1

        # Accumulate batch statistics
        batch_reward += _reward
        batch_loss -= _reward * log_prob  # Policy gradient loss

    return cids, batch_reward, batch_loss


def policy_gradient_train(policy_model, problems, train_pids, cand_pids, cand_examples, args):
    """
    Train the policy network using policy gradient (REINFORCE algorithm).
    
    Args:
        policy_model: Neural network that learns to select few-shot examples
        problems: Dictionary of all problems
        train_pids: Problem IDs for training
        cand_pids: Problem IDs for candidate examples
        cand_examples: Pre-processed candidate examples
        args: Command line arguments
    """
    # Initialize optimizer
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    # Prepare training data
    train_samples, train_labels, units, options = [], [], [], []
    for pid in train_pids:
        train_samples.append(create_example_from_pid(pid, problems, args, test=True))
        answer_norm = normalize_answer(problems[pid]['answer'], problems[pid]['unit'])
        train_labels.append(answer_norm)
        units.append(problems[pid]['unit'])
        options.append(problems[pid]['choices'])

    num_batch = math.ceil(len(train_samples) / args.batch_size)

    # Initialize tracking variables
    reward_history = []
    loss_history = []
    total_reward_history = []
    total_loss_history = []
    STOP_FLAG = False

    # Training loop
    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # Process batches
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            
            # Prepare batch data
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = train_labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            pid_batch = train_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            unit_batch = units[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            option_batch = options[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]

            # Get embeddings from policy network
            embedding_cands = policy_model(cand_examples)
            embedding_ctxt = policy_model(train_batch)

            # Calculate similarity scores
            scores = torch.mm(embedding_ctxt, embedding_cands.t())
            scores = F.softmax(scores, dim=1)

            # Get rewards and loss
            cids, reward, loss = get_batch_reward_loss(scores, cand_pids, pid_batch, option_batch, unit_batch,
                                                       label_batch, args)

            # Log batch statistics
            logger.write(f"cids for sample[-1] in batch: {cids}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # Update policy network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_train_reward += reward
            total_train_loss += loss.item()
            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # Update epoch statistics
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        # Find best performance
        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)
        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        # Log epoch statistics
        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # Save checkpoints
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # Save best models
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # Save training history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # Log cache info
        logger.write(call_gpt3.cache_info())
        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # Save final model
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    """
    Parse command line arguments.
    Returns parsed arguments with default values.
    """
    parser = argparse.ArgumentParser()
    
    # Data paths and model settings
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # User options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--train_number', type=int, default=20)
    parser.add_argument('--cand_number', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')

    args = parser.parse_args()

    # Create checkpoint directory and save arguments
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Load data
    problems, cand_pids, train_pids = load_data(args)

    # Initialize policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    # Move model to GPU if available
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(device)

    # Prepare candidate examples
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)

    # Start training
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(policy_model, problems, train_pids, cand_pids, cand_examples, args) 