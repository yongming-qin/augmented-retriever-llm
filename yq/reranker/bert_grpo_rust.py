"""
Use Bert and one linear layer to generate a embedding for the query and the retrieved docs.
Use the embedding to calculate the similarity between the query and the BM25 retrieved docs.

Yongming
2025-06-24
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import dotenv
from eval_rag_rq4 import get_code_generation_prompt_rag, call_llm, extract_rust_code, create_test_file, run_rust_test


# Define BERT Embedder with frozen BERT and trainable linear layer
class BERTEmbedder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', output_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        # The input_ids are the tokenized and encoded numerical IDs that represent your input text.
        # The attention_mask is a tensor that tells the model which tokens are actual input and which ones are padding.
        # The token_type_ids (also called segment IDs) are used in BERT-style models to distinguish between different segments in a single input — typically two sentences.
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.linear(cls_embedding), dim=-1)
    

class GrpoLearning():
    def __init__(self, query_items, api_items):
        # query_items and api_items are both lists of dicts
        self.model = BERTEmbedder(output_dim=256)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.optimizer =  torch.optim.Adam(self.model.linear.parameters(), lr=1e-4)
        
        # Data setup
        self.query_items = query_items
        self.idxs = [item["task_idx"] for item in query_items]
        self.query_texts = [item["query"] for item in query_items]
        self.api_texts = [str(item) for item in api_items]
        self.bm25 = BM25Okapi([text.split() for text in self.api_texts])
        
        ## LLM setup
        dotenv.load_dotenv()
        self.llm_model = "gpt-4.1-nano"  # Renamed from self.model to self.llm_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")

    # GRPO loss
    def grpo_loss(self, current_probs, old_probs, advantages, epsilon=0.2):
        ratio = current_probs / (old_probs + 1e-8)
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        return -torch.mean(torch.min(ratio * advantages, clipped * advantages))

    # Simulate reward (compiler-based)
    def simulate_reward(self, chunk):
        return random.choice([0.0, 0.5, 1.0])  # Replace with real compiler check
    
    # Tokenization helper
    def tokenize_texts(self, texts, max_length=256):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    # Training loop
    def train(self, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for query, top_docs in self.dataloader:
                self.optimizer.zero_grad()

                # Tokenize query and docs. need to be a list. even one batch.
                query_tokens = self.tokenize_texts(list(query))
                doc_tokens = self.tokenize_texts(list(top_docs))

                # Compute embeddings
                with torch.no_grad():
                    query_emb = self.model(**query_tokens)  # [1, dim]
                doc_embs = self.model(**doc_tokens)         # [10, dim]

                # Similarity and softmax
                sims = F.cosine_similarity(query_emb, doc_embs)
                probs = F.softmax(sims, dim=0)
                old_probs = probs.detach()

                # Reward and advantage
                rewards = torch.tensor([self.simulate_reward(c) for c in top_docs[0]])
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                # GRPO loss
                loss = self.grpo_loss(probs, old_probs, advantages)
                loss.backward()
                self.optimizer.step()

                print(f"Query: {query[0]} | Loss: {loss.item():.4f}")

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def test_one_query(self, item_idx):
        # Extract related information
        query_text = self.query_texts[item_idx]
        function_signature = self.query_items[item_idx]["function_signature"]
        test_program = self.query_items[item_idx]["test_program"]
        
        # Retrieve relevant documentation using BM25
        # Use get_scores instead to get_top_n for more control.
        scores = self.bm25.get_scores(query_text.split())  # List of scores for all db_chunks
        top_n = 10
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        bm25_top_docs = [self.api_texts[i] for i in top_indices]
        print(f"Retrieved docs indices: {top_indices}")
        
        ## Run model and calculate the similarity between the query and the retrieved docs        
        query_tokens = self.tokenize_texts(list(query_text)) # need to be a list. even one batch.
        doc_tokens = self.tokenize_texts(list(bm25_top_docs)) # several batches.
        with torch.no_grad():
            q_emb = self.model(**query_tokens)
            d_embs = self.model(**doc_tokens)
            sims = F.cosine_similarity(q_emb, d_embs)
            
        sorted_sims = np.argsort(sims.numpy())[::-1]
        model_top_indices = [top_indices[i] for i in sorted_sims]
        
        print(f"BM25 ranking: {top_indices}")
        print(f"Model ranking: {model_top_indices}")

        retrieve_k = 4
        retrieved_docs = [self.api_texts[i] for i in model_top_indices[:retrieve_k]]
        prompt_rag = get_code_generation_prompt_rag(query_text, retrieved_docs, function_signature)
        print(prompt_rag)
        
        ## Call llm to generate the code
        raw_response = call_llm(prompt_rag, self.llm_model, self.api_key, self.base_url)
        code = extract_rust_code(raw_response)
        print(f"llm generated code: {code}\n--------------------------------\n")
        
        # Determine Rust version to use based on task information
        rust_version = "1.84.0"  # Default version
        #####################TODO: research for different Embedders for different rust version
        # Run test to validate the solution
        test_file = create_test_file(code, test_program)
        success, error_message = run_rust_test(test_file, rust_version)
        print(f"test result: {success=}, compilation error: {error_message=}\n")
        
        return



def main():
    with open("RustEvo^2.json", "r", encoding="utf-8") as f:
        query_data = json.load(f)
    query_items = [item for item in query_data] # each item is a dict, check the json file for more details
    
    with open("APIDocs.json", "r", encoding="utf-8") as f:
        api_data = json.load(f)
    api_items = [item for item in api_data] # each item is a dict, check the json file for more details
    
       
    grpo_learning = GrpoLearning(query_items, api_items)
    # Test
    grpo_learning.test_one_query(0)
    
    return
    # Train
    grpo_learning.train(epochs=3)
    grpo_learning.save_model("grpo_model.pth")
    grpo_learning.test_one_query(0)


if __name__ == "__main__":
    main()