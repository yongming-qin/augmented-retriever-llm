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

from dotenv import load_dotenv
from together import Together

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from transformers import get_scheduler


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
    
    problems_train = json.load(open(path, 'r'))
    print(f"\nLoaded {len(problems_train)} samples from PubMedQA dataset")
    
    problmes_id = list(problems_train.keys())
    
    return problems_train, problmes_id




openai_api_key = os.getenv('OPENAI_API_KEY')
def process_with_gpt4(question):
    """
    Process a question using GPT-4 API.
    """
    client = openai.OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful medical research assistant. Answer the following medical question based on your knowledge."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

together_api_key = os.getenv('TOGETHER_API_KEY')
def process_with_together(question):
    """
    Process a question using Together.ai API.
    """
    client = Together(api_key=together_api_key)
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical research assistant. Answer the following medical question based on your knowledge."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing question: {e}")
        return None
    



def check_hallucination(gpt4_answer, original_answer):
    """
    Use openai to check if the answer from GPT-4 is hallucinated.
    """
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": f"The question is: {question}. Is the answer from llm hallucinated in short answer? The llm answer is: {gpt4_answer}. The original answer is: {original_answer}."}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


def process_with_fewshot(question, problems_id, problems_train, n=2):
    """
    Process a question using few-shot prompting.
    """
    prompt = f"The question is: {question}. The original answer is: {problems_train[problems_id[i]]['LONG_ANSWER']}. Please answer the question based on the original answer."


if __name__ == '__main__':
    ###############################################
    ############### Load data #####################
    ###############################################
    # load_data_kilt('/home/yq/ssd/hallucination/kilt-meta/nq-train-kilt.jsonl')
    
    # path_train_pubmed = '/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqaa_train_set.json'
    path_train_pubmed = '/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqal_fold0/train_set.json'
    problems_train, problems_id = load_data_pubmedqa(path_train_pubmed)
    
    # Get OpenAI API key from .env file

    load_dotenv()
    
    ###############################################
    ############### Process data ##################
    ###############################################
    # Process a few sample questions
    print("\nAnswering questions with LLM and checking hallucination with OpenAI.")
    question_list = [3, 368]
    for i in question_list:
        question = problems_train[problems_id[i]]['QUESTION']
        # llm_answer = process_with_gpt4(question)
        llm_answer = process_with_together(question)
        # Build the output text
        output_text = f"\nQuestion {i}: {question}\n\n"
        output_text += "----------------------------------------------"
        output_text += f"LLM Answer: {llm_answer}\n\n"
        output_text += "----------------------------------------------"
        output_text += f"Original Answer: {problems_train[problems_id[i]]['LONG_ANSWER']}\n\n"
        output_text += "----------------------------------------------"
        hallucination_check = check_hallucination(llm_answer, problems_train[problems_id[i]]['LONG_ANSWER'])
        output_text += f"Hallucination Check: {hallucination_check}\n\n"
        output_text += "----------------------------------------------"
        our_prompt = prompt_with_fewshot(question, problems_id, problems_train, n=2)
        our_answer = process_with_together(our_prompt)
        output_text += f"Fewshot Answer: {our_answer}\n\n"
        output_text += "----------------------------------------------"
        output_text += f"==============================================\n\n"
        
        # Write to file and print to console
        with open('result.txt', 'a') as f:
            f.write(output_text)
            f.flush() # Ensure immediate write to disk
            
        print(output_text)
        
        time.sleep(1)  # Add a small delay to avoid rate limiting

    ###############################################
    ############### Train model ###################
    ###############################################
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True
    
    # Initialize policy network
    # Initialize BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=1)
    
    # Move model to GPU if available
    model.to(torch.device("cuda"))
    
    # Freeze the BERT encoder
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3}
    ])
    
    
    # A linear warmup + decay scheduler
    total_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
