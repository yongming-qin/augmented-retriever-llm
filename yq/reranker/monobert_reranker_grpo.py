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

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import random
import json
from collections import defaultdict
from tqdm import tqdm
from rag_dataset import PubmedqaDataset, CompareRag

class GrpoLearningMonoBert(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) #YQ 

        self.data = dataset.data
        self.bm25 = dataset.bm25
        self.bm25_corpus = dataset.bm25_corpus
        self.compare_rag = CompareRag(dataset)

    def compute_rewards_batch(self, questions, answers, contexts_batch):
        rewards = []
        for question, answer, contexts in zip(questions, answers, contexts_batch):
            question_rewards = []
            for idx, ctx in enumerate(contexts):
                if idx == 0:
                    llm_answer = self.compare_rag.no_rag_answer(question)
                else:
                    llm_answer = self.compare_rag.gt_answer(question, ctx)
                reward = self.compare_rag.compute_reward(question, llm_answer, answer)
                question_rewards.append(reward)
            rewards.append(question_rewards)
        return torch.tensor(rewards)

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

                    questions.append(q)
                    answers.append(a)
                    all_contexts_batch.append(contexts)
                    pair_batch.extend(pairs)  # Flatten the pairs list

                # Tokenize all question-context pairs
                tokenized = self.tokenizer(pair_batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**tokenized)
                logits = outputs.logits.squeeze(-1)
                print(f"logits: {logits}")
                
                # Prepare targets (rewards)
                rewards = self.compute_rewards_batch(questions, answers, all_contexts_batch)
                norm_rewards = rewards / 100.0
                print(f"norm_rewards: {norm_rewards}")
                
                
                # Reshape logits to match rewards shape
                num_contexts = len(all_contexts_batch[0])  # Number of contexts per question
                logits_reshaped = logits.view(len(questions), num_contexts)
                
                loss = F.mse_loss(logits_reshaped, norm_rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"[{batch_start}] Batch Loss: {loss.item():.4f}")

    def evaluate(self):
        self.compare_rag.test_rag(indices=range(20, 30))



def main():
    
    dataset = PubmedqaDataset()
    grpo = GrpoLearningMonoBert(dataset)
    grpo.train(epochs=3, top_n=1, batch_size=2)
    # grpo.evaluate()


if __name__ == "__main__":
    main()
