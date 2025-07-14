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

v3: after trying the mse regression wrt the hallucination score, I found that the model is not able to predict the score well.
So I try to use the 3-class classifier to predict the class of the hallucination score.
Still, the model is not able to predict the class well.
But I found that the margin_ranking_loss between two paris (question, context) is more stable.
Now, I try to use the margin_ranking_loss to train the model.

SELF: 
1. add the "I AM NOT SURE" to the contexts.
2. add the reward_i_not_sure to the rewards.
3. use the grpo loss to train the model.

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


class CrossEncoderWithSigmoid(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **inputs):
        logits = self.base_model(**inputs).logits
        probs = self.sigmoid(logits)  # restrict output to [0, 1]
        return probs

class GrpoLearningMonoBert(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # Load or init cache for cachine the llm answer and reward
        self.cache_path = "reward_cache.json"
        if os.path.exists(self.cache_path):
            self.load_cache()
        else:
            self.reward_cache = {}

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = CrossEncoderWithSigmoid(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) #YQ 
        
        print(self.model)

        #YQ: freeze the base model
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        self.data = dataset.data
        self.bm25 = dataset.bm25
        self.bm25_corpus = dataset.bm25_corpus
        self.compare_rag = CompareRag(dataset)
        
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "checkpoint.pth")
        print(f"Checkpoint saved")
        
    def load_checkpoint(self):
        self.model.load_state_dict(torch.load("checkpoint.pth"))
        print(f"Checkpoint loaded")
        
    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.reward_cache, f)
        print(f"Saved {len(self.reward_cache)} rewards to cache")

    def load_cache(self):
        with open(self.cache_path, "r") as f:
            self.reward_cache = json.load(f)
        print(f"Loaded {len(self.reward_cache)} rewards from cache")


    def _cache_key(self, question, context):
        return f"{question} ||| {context}"
    
    def reward_i_not_sure(self, question_rewards):
        # question_rewards[0] = reward of (question, "NO CONTEXT IS NEEDED")
        # question_rewards[1] = reward of (question, gt_context)
        # question_rewards[2:] = reward of (question, retrieved_contexts)
        if np.max(question_rewards) < 50:
            reward = 100
        else:
            reward = 0
        return reward

    def compute_rewards_batch(self, questions, answers, contexts_batch):
        rewards = []
        for question, answer, contexts in zip(questions, answers, contexts_batch):
            question_rewards = []
            for ctx in contexts:
                if ctx == "I AM NOT SURE":
                    reward = self.reward_i_not_sure(question_rewards)
                    question_rewards.append(reward)
                    
                    #YQ: adjust the reward of NO CONTEXT IS NEEDED
                    if question_rewards[0] >= 80:
                        question_rewards[0] = 100
                    elif question_rewards[1] - question_rewards[0] >= 30:
                        question_rewards[0] = 0
                    continue #YQ: actually this is already the last one.
                
                key = self._cache_key(question, ctx)
                if key in self.reward_cache:
                    reward = self.reward_cache[key]
                else:
                    llm_answer = self.compare_rag.gt_answer(question, ctx)
                    reward = self.compare_rag.compute_reward(question, llm_answer, answer)
                    self.reward_cache[key] = reward
                question_rewards.append(reward)
            rewards.extend(question_rewards)
        return torch.tensor(rewards) / 100.

    def train(self, epochs, top_n, batch_size):
        self.batch_size = batch_size
        self.model.train()
        keys = list(self.data.keys())

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            random.seed(epoch)
            random.shuffle(keys)

            step = 0
            for batch_start in tqdm(range(0, len(keys), batch_size), desc=f"Epoch {epoch + 1}"):
                batch_keys = keys[batch_start:batch_start + batch_size]
                questions, answers, all_contexts_batch, pair_batch = [], [], [], []

                for key in batch_keys:
                    q = self.data[key]["question"]
                    a = self.data[key]["answer"]
                    gt_ctx = self.data[key]["context"]
                    retrieved_ctxs = self.bm25.get_top_n(q.split(), self.bm25_corpus, n=top_n)
                    ##YQ: each retrieved context is just a string, ensured by the code.
                    contexts = ["NO CONTEXT IS NEEDED", gt_ctx] + retrieved_ctxs + ["I AM NOT SURE"]
                    pairs = [f"{q} [SEP] {ctx}" for ctx in contexts]

                    questions.append(q)
                    answers.append(a)
                    all_contexts_batch.append(contexts)
                    pair_batch.extend(pairs)  # Flatten the pairs list, [n_questions * n_contexts]

                # Tokenize all question-context pairs
                tokenized = self.tokenizer(pair_batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**tokenized)
                logits = outputs.logits.squeeze(-1) # [n_questions * n_contexts]
                
                # Prepare targets (rewards), [n_questions * n_contexts]
                rewards = self.compute_rewards_batch(questions, answers, all_contexts_batch)
                
                # Reshape logits to match rewards shape
                logits_batch = logits.view(self.batch_size, -1)
                rewards_batch = rewards.view(self.batch_size, -1)

                for i in range(self.batch_size):
                    logits_row = " ".join(f"{x:.2f}" for x in logits_batch[i])
                    rewards_row = " ".join(f"{x:.2f}" for x in rewards_batch[i])
                    print(f"[Sample {i}] Logits:  {logits_row}")
                    print(f"[Sample {i}] Rewards: {rewards_row}")

                
                # loss = self.grpo_loss(logits, rewards)
                # loss = self.margin_loss(logits, rewards)
                loss = self.regression_loss(logits, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"[{batch_start}] Batch Loss: {loss.item():.4f}")
                step += 1
                if step % 100 == 0:
                    self.save_checkpoint()

    def grpo_loss(self, logits, rewards, temperature=1.0):
        logits = logits.view(self.batch_size, -1)
        rewards = rewards.view(self.batch_size, -1)
        # scores: [batch_size, num_ctx]
        # rewards: [batch_size, num_ctx]
        scores = logits / temperature
        rewards = rewards / temperature

        pred_probs = F.log_softmax(scores, dim=1)         # [B, N]
        # target_probs = F.softmax(rewards, dim=1)          # [B, N]
        loss = -(rewards * pred_probs).sum(dim=1)    # [B]
        return loss.mean()
    
    def margin_loss(self, logits, rewards):
        logits = logits.view(-1, 2)
        rewards = rewards.view(-1, 2)
        score_i = logits[:, 0]
        score_j = logits[:, 1]
        target = (rewards[:, 0] >= rewards[:, 1]).float() * 2 - 1  # maps True→1, False→-1
        return F.margin_ranking_loss(score_i, score_j, target, margin=0.1)

    def regression_loss(self, logits, rewards):
        loss = F.mse_loss(logits, rewards)
        return loss
        

    def test_rag(self):
        self.compare_rag.test_rag(indices=range(20, 30))
        
        
def evaluate(self):
    dataset = PubmedqaDataset(raw_data_path="/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqal_fold0/dev_set.json")
    grpo = GrpoLearningMonoBert(dataset)
    
    
    

def main():
    dataset = PubmedqaDataset(raw_data_path="/home/yq/ssd/hallucination/augmented-retriever-llm/cluster_results/pubmed/pqal_fold0/train_set.json")
    grpo = GrpoLearningMonoBert(dataset)
    # grpo.test_rag()

    try:
        grpo.train(epochs=3, top_n=2, batch_size=2)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving cache...")
    finally:
        grpo.save_cache()
        print("Cache saved.")

if __name__ == "__main__":
    main()
