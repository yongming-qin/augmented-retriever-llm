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
import datetime
from dotenv import load_dotenv
from together import Together



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

def load_data_kilt(path):
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


def load_data_pubmedqa(path):
    """
    Load and process the PubMedQA dataset.
    Returns a list of dictionaries containing question-answer pairs.
    """
    
    train_data = json.load(open(path, 'r'))
    print(f"\nLoaded {len(train_data)} samples from PubMedQA dataset")
    train_keys = list(train_data.keys())
    
    return train_data, train_keys




openai_api_key = os.getenv('OPENAI_API_KEY')
def process_with_gpt4(model, messages):
    """
    Process a question using GPT-4 API.
    """
    client = openai.OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model= model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content, response.usage
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

together_api_key = os.getenv('TOGETHER_API_KEY')
def process_with_together(model, messages):
    """
    Process a question using Together.ai API.
    """
    client = Together(api_key=together_api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content, response.usage
    except Exception as e:
        print(f"Error processing question: {e}")
        return None
    
def check_hallucination_gpt4(model, question, llm_answer, original_answer):
    """
    Use GPT-4 to check if the answer is hallucinated.
    """
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"The question is: {question}. Is the answer from llm hallucinated? First give explanation, then answer from three options: hallucinated, not hallucinated, and not sure. The llm answer is: {llm_answer}. The original answer is: {original_answer}."}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content, response.usage


def check_hallucination_together(model, question, llm_answer, original_answer):
    """
    Use Together.ai to check if the answer is hallucinated.
    """
    client = Together(api_key=together_api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"The question is: {question}. Is the answer from llm hallucinated? First give explanation, then answer from three options: hallucinated, not hallucinated, and not sure. The llm answer is: {llm_answer}. The original answer is: {original_answer}."}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content, response.usage
    except Exception as e:
        print(f"Error checking hallucination: {e}")
        return None, None

def check_hallucination(model, question, gpt4_answer, original_answer):
    """
    Check if the answer is hallucinated.
    """
    if "gpt" in model:
        return check_hallucination_gpt4(model, question, gpt4_answer, original_answer)
    elif "llama" in model:
        return check_hallucination_together(model, question, gpt4_answer, original_answer)
    else:
        raise ValueError(f"Model {model} is not supported.")

def create_fewshot_messages(question, train_data, train_keys, n=2):
    """
    Create few-shot messages for the question.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the following questions based on your knowledge."},
    ]
    keys = []
    for _ in range(n):
        key = random.choice(train_keys)
        messages.append({"role": "user", "content": f"Question: {train_data[key]['QUESTION']}\nAnswer: {train_data[key]['LONG_ANSWER']}"})
        keys.append(key)
    messages.append({"role": "user", "content": f"Question: {question}\nAnswer: "})
    return messages, keys

if __name__ == '__main__':
    # load_data_kilt('/home/yq/ssd/hallucination/kilt-meta/nq-train-kilt.jsonl')
    
    # path_train_pubmed = '/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqaa_train_set.json'
    path_train_pubmed = '/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqal_fold0/train_set.json'
    train_data, train_keys = load_data_pubmedqa(path_train_pubmed)
    
    # Get OpenAI API key from .env file
    load_dotenv()
    
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True
    
    # Process a few sample questions
    model_answering = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # model_answering = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    model_checking_hallucination = "gpt-4.1-nano"
    # model_checking_hallucination = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    print(f"\nAnswering questions with {model_answering} and checking hallucination with {model_checking_hallucination}.")
    
    # clear the result file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"{current_time}_result_{model_answering.replace('/', '_')}_{model_checking_hallucination.replace('/', '_')}.txt"
    with open(file_name, 'w') as f:
        f.write(f"Answering questions with {model_answering} and checking hallucination with {model_checking_hallucination}.\n\n")
        
        
    data_sample = [3, 368]
    for i in data_sample:
        question = train_data[train_keys[i]]['QUESTION']
        long_answer = train_data[train_keys[i]]['LONG_ANSWER']
        
        
        zero_shot_messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the following questions based on your knowledge."},
            {"role": "user", "content": question}
        ]
        llm_answer, llm_usage = process_with_together(model=model_answering, messages=zero_shot_messages)
        # Build the output text
        output_text = f"\nQuestion {i}: {question}\n\n"
        output_text += "----------------------------------------------\n"
        output_text += f"==LLM Answer== (cost: {llm_usage.total_tokens} tokens): {llm_answer}\n\n"
        output_text += "----------------------------------------------\n"
        output_text += f"==Original Answer==: {long_answer}\n\n"
        output_text += "----------------------------------------------\n"
        # check hallucination
        hallucination_check, hallucination_usage = check_hallucination(model=model_checking_hallucination, question=question, gpt4_answer=llm_answer, original_answer=long_answer)
        output_text += f"==Hallucination Check== (cost: {hallucination_usage.total_tokens} tokens): {hallucination_check}\n\n"
        output_text += "----------------------------------------------\n"
        
        
        # in-context learning
        fewshot_messages, fewshot_keys = create_fewshot_messages(question, train_data, train_keys, n=2)
        fewshot_answer, fewshot_usage = process_with_together(model=model_answering, messages=fewshot_messages)
        output_text += f"==Fewshot Answer== (cost: {fewshot_usage.total_tokens} tokens): {fewshot_answer}\n\n"
        output_text += "----------------------------------------------\n"
        # check hallucination
        hallucination2_check, hallucination2_usage = check_hallucination(model=model_checking_hallucination, question=question, gpt4_answer=fewshot_answer, original_answer=long_answer)
        output_text += f"==Hallucination2 Check== (cost: {hallucination2_usage.total_tokens} tokens): {hallucination2_check}\n\n"
        output_text += f"===============total cost===============\n\n"
        output_text += f"data sample {train_keys[i]} total cost: {llm_usage.total_tokens + fewshot_usage.total_tokens + hallucination_usage.total_tokens + hallucination2_usage.total_tokens} tokens. keys: {fewshot_keys}\n\n"
        
        
        # Write to file and print to console
        with open(file_name, 'a') as f:
            f.write(output_text)
            f.flush() # Ensure immediate write to disk

        print(f"==Hallucination Check== (cost: {hallucination_usage.total_tokens} tokens): {hallucination_check}\n\n")
        print(f"==Hallucination2 Check== (cost: {hallucination2_usage.total_tokens} tokens): {hallucination2_check}\n\n")
        print(f"data sample {train_keys[i]} total cost: {llm_usage.total_tokens + fewshot_usage.total_tokens + hallucination_usage.total_tokens + hallucination2_usage.total_tokens} tokens\n\n")
        print(f"keys: {fewshot_keys}\n\n")
        
        time.sleep(1)  # Add a small delay to avoid rate limiting

    
