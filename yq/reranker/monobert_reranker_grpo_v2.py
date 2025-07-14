""" Full implementation using pretrained MonoBERT for reranking and GRPO-style training
This file uses the pubmedqa dataset for training and testing.

The training process is as follows:
1. Load the pubmedqa dataset.
2. Load the pretrained MonoBERT model and tokenizer.
3. Tokenize the query and the context.
4. Pass the query and the context to the MonoBERT model to get the logits.
5. Use the logits to rerank the context.
6. Use the reranked context to generate the code.
7. Test the code.

Yongming, Chenxi
2025-07-07

v2: to predict 3 classes
✅ Goal:
Convert hallucination scores (0–100) into discrete classes:

0–30 → class 0 (high hallucination)

30–70 → class 1 (medium)

70–100 → class 2 (low hallucination)

You’ll train a 3-class classifier using CrossEntropyLoss, predicting which hallucination bin the (question, context) pair falls into.



"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

import numpy as np
import random
import json
from collections import defaultdict
from tqdm import tqdm
from rag_dataset import PubmedqaDataset, CompareRag

class GrpoLearningMonoBert(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # Load or init cache for cachine the llm answer and reward
        if os.path.exists("reward_cache.json"):
            self.load_cache()
        else:
            self.reward_cache = {}

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)
        
        print(self.model)

        #YQ: freeze the base model
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5) #YQ 

        self.data = dataset.data
        self.bm25 = dataset.bm25
        self.bm25_corpus = dataset.bm25_corpus
        self.compare_rag = CompareRag(dataset)
        
    def save_cache(self, path="reward_cache.json"):
        with open(path, "w") as f:
            json.dump(self.reward_cache, f)

    def load_cache(self, path="reward_cache.json"):
        with open(path, "r") as f:
            self.reward_cache = json.load(f)


    def _cache_key(self, question, context):
        return f"{question} ||| {context}"

    def compute_rewards_batch(self, questions, answers, contexts_batch):
        rewards = []
        for question, answer, contexts in zip(questions, answers, contexts_batch):
            question_rewards = []
            for ctx in contexts:
                key = self._cache_key(question, ctx)
                if key in self.reward_cache:
                    reward = self.reward_cache[key]
                else:
                    llm_answer = self.compare_rag.gt_answer(question, ctx)
                    reward = self.compare_rag.compute_reward(question, llm_answer, answer)
                    self.reward_cache[key] = reward
                question_rewards.append(reward)
            rewards.extend(question_rewards)
            
        # Convert rewards to classes
        bins = torch.tensor([30, 70])  # boundaries
        # Bucketize gives: 0 for <=30, 1 for (30,70], 2 for >70
        classes = torch.bucketize(torch.tensor(rewards), boundaries=bins, right=True)
        return classes

    def train(self, epochs=3, top_n=3, batch_size=4):
        self.model.train()
        keys = list(self.data.keys())

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            random.shuffle(keys)

            for batch_start in tqdm(range(0, len(keys), batch_size), desc=f"Epoch {epoch + 1}"):
                batch_keys = keys[batch_start:batch_start + batch_size]
                questions, answers, all_contexts_batch, pair_batch = [], [], [], []

                for key in batch_keys:
                    q = self.data[key]["question"]
                    a = self.data[key]["answer"]
                    gt_ctx = self.data[key]["context"]
                    retrieved_ctxs = self.bm25.get_top_n(q.split(), self.bm25_corpus, n=top_n)
                    ##YQ: each retrieved context is just a string, ensured by the code.
                    contexts = ["NO CONTEXT IS NEEDED", gt_ctx] + retrieved_ctxs
                    pairs = [f"{q} [SEP] {ctx}" for ctx in contexts]
                    print(f"len of pairs: {len(pairs)}")

                    questions.append(q)
                    answers.append(a)
                    all_contexts_batch.append(contexts)
                    pair_batch.extend(pairs)  # Flatten the pairs list, [n_questions * n_contexts]

                # Tokenize all question-context pairs
                tokenized = self.tokenizer(pair_batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**tokenized)
                logits = outputs.logits # [n_questions * n_contexts, 3]
                
                # Prepare targets (rewards), [n_questions * n_contexts]
                rewards = self.compute_rewards_batch(questions, answers, all_contexts_batch)
                
                # Reshape logits to match rewards shape
                print(f"logits: {logits}")
                print(f"rewards: {rewards}")
                
                loss = F.cross_entropy(logits, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"[{batch_start}] Batch Loss: {loss.item():.4f}")

    def loss(self, logits, rewards):
        
    
        def margin_loss(logits, rewards):
            score_i = logits[:, 0]
            score_j = logits[:, 1]
            target = (rewards[:, 0] >= rewards[:, 1]).float() * 2 - 1  # maps True→1, False→-1
            return F.margin_ranking_loss(score_i, score_j, target, margin=0.1)

        def regression_loss(logits, rewards):
            loss = F.mse_loss(logits, rewards)
            return loss
        
        return margin_loss(logits, rewards)

    def evaluate(self):
        self.compare_rag.test_rag(indices=range(20, 30))
        

def main():
    
    dataset = PubmedqaDataset()
    grpo = GrpoLearningMonoBert(dataset)
    # grpo.evaluate()

    try:
        grpo.train(epochs=3, top_n=1, batch_size=1)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving cache...")
    finally:
        grpo.save_cache()
        print("Cache saved.")

if __name__ == "__main__":
    main()
