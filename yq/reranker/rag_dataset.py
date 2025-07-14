"""


"""


import os
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import re
import sys
sys.path.append('/home/yq/ssd/hallucination/RustEvo/yq')
from chat_models import LmClient



"""
Read the PubMedQA dataset.
self.keys is a list of question ids.
self.questions is a list of questions.
self.answers is a list of answers.
self.bm25 is a BM25Okapi object.
self.bm25_corpus is used for BM25 retrieval. It is a list of strings, each string is a context.
"""
class PubmedqaDataset:
    def __init__(self, raw_data_path):
        raw_data = json.load(open(raw_data_path, 'r'))
        print(f"\nLoaded {len(raw_data)} samples from PubMedQA dataset")
        
        self.data = {}
        for key, value in raw_data.items():
            self.data[key] = {
                "question": value["QUESTION"],
                "answer": value["LONG_ANSWER"],
                "context": "\n".join(value["CONTEXTS"])
            }
        
        # context is a list of paragraphs, each paragraph is a string
        
        # Each context is a list of string.
        self.bm25_corpus = ["\n".join(value["CONTEXTS"]) for value in raw_data.values()]
        bm25_words = [context.split() for context in self.bm25_corpus]
        self.bm25 = BM25Okapi(bm25_words) # Does not include the corpus.



class NqDataset:
    def __init__(self):
        # The path to the NQ dataset.
        path = "/home/yq/ssd/hallucination/kilt-meta/KILT/data/nq-train-kilt.jsonl"
        
        # The path to the Wikipedia database.
        wikipedia_database_path = "/home/yq/ssd/hallucination/kilt-meta/kilt_wikipedia_subset_nq.jsonl"
        wikipedia_database = json.load(open(wikipedia_database_path, 'r'))
        wikipedia_dict = {item["wikipedia_id"]: "\n".join(item["text"]) for item in wikipedia_database}
        
        self.data = {}
        with open(path, 'r') as f:
            print(f"\nLoaded {len(f)} lines from NQ dataset")
            for line in f:
                sample = json.loads(line.strip())
                outputs = sample["output"]
                if len(outputs) == 0:
                    raise ValueError(f"No outputs found for sample {sample['id']}")
                outputs.sort(key=lambda x: x["provenance"]["bleu_score"] if "answer" in x and "provenance" in x and "bleu_score" in x["provenance"] else 0, reverse=True)
                best_answer = outputs[0]["answer"]
                best_wikipedia_id = outputs[0]["provenance"]["wikipedia_id"]
                    
                self.data[sample["id"]] = {
                    "question": sample["input"],
                    "answer": best_answer,
                    "context": wikipedia_dict[best_wikipedia_id]
                }

    
    def print_sample(self):
        # Print some sample data
        print(f"\nLoaded {len(self.data)} samples from KILT Natural Questions dataset")
        print("\nSample 1:")
        print(f"Question: {self.data[0]['input']}")
        print(f"Answer: {self.data[0]['answer']}")
        print(f"Provenance: {self.data[0]['context']}")

    


"""
no rag
bm25 rag
gt
"""
class CompareRag:
    def __init__(self, dataset):
        self.dataset = dataset
        
        ## load the model
        # lm_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        # lm_model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
        # lm_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        # lm_model = "claude-3-5-haiku-20241022"
        # "gpt-4.1-nano"
        lm_model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
        # lm_model = "mistralai/Mistral-7B-Instruct-v0.3"
        lm_hallucination_model = "gpt-4.1-nano"
        
        self.lm_answer = LmClient(model=lm_model, temperature=0.0)
        self.lm_hallucination = LmClient(model=lm_hallucination_model, temperature=0.0)
        
    def prompt_query(self, question, context):
        query = f"Answer the question based on the following context in less than 100 words:\n{context}\n\nQuestion: {question}"
        return query
        
    def no_rag_answer(self, query):
        messages = [{"role": "user", "content": query}]
        response = self.lm_answer.chat(messages=messages)
        # print(f"query: {query} Answer: {response}")
        if len(response) < 20:
            raise ValueError(f"LLM response is too short: {response}")
        return response

    def gt_answer(self, query, context):
        if context == "NO CONTEXT IS NEEDED":
            return self.no_rag_answer(query)
        else:
            full_query = self.prompt_query(query, context)
            return self.no_rag_answer(full_query)

    def bm25_answer(self, question, bm25, bm25_corpus, top_k=3):
        retrieved = bm25.get_top_n(question.split(), bm25_corpus, n=top_k) # is a list of list of strings
        # Flatten the retrieved contexts (each context is a list of strings)
        flattened_contexts = []
        for context_list in retrieved:
            flattened_contexts.extend(context_list)
        context = "\n".join(flattened_contexts)
        full_query = self.prompt_query(question, context)
        return self.no_rag_answer(full_query)
    

    def check_hallucination(self, question, llm_answer, original_answer):
        # Check if the LLM answer is hallucinated
        messages = [{
            "role": "user",
            "content": (
                f"Given the following question, the LLM's answer, and the reference answer, assess whether the LLM's response is correct and not hallucinated.\n\n"
                f"Question: {question}\n"
                f"LLM Answer: {llm_answer}\n"
                f"Reference Answer: {original_answer}\n\n"
                "Step 1: Provide a brief explanation of your reasoning based on the comparison between the LLM answer and the reference.\n"
                "Step 2: Make a final judgment from the following three options:\n"
                " - hallucinated\n"
                " - not hallucinated\n"
                " - not sure\n\n"
                "Step 3: Provide a score between 0 and 100 for the LLM's answer, where 0 is the worst and 100 is the best.\n\n"
                "IMPORTANT: Return the final answer in the following format exactly:\n"
                "Explanation: <your explanation here>\n"
                "Judgment: <one of: hallucinated | not hallucinated | not sure>\n"
                "Score: <an integer between 0 and 100>\n"
                "Only output the fields in this order and format. Do not include any other text."
            )
        }]

        
        response = self.lm_hallucination.chat(messages=messages)
        if len(response) < 20:
            raise ValueError(f"LLM response is too short: {response}")
        return response
    
    def compute_reward(self, question, llm_answer, original_answer):
        check_hallucination_response = self.check_hallucination(question, llm_answer, original_answer)
        # check_hallucination_response is a string from the LLM like:
        """Hallucination (GT): Explanation: The LLM's answer provides specific comparative data on continuing oral hypoglycemic agents with insulin versus insulin monotherapy, including outcomes like HbA1c improvement, treatment failures, weight gain, and hypoglycemic events. The reference answer is more general, stating that adding bedtime NPH insulin to maximal sulfonylurea and metformin therapy is effective and well-tolerated. The LLM's details are consistent with the general principle in the reference that oral agents are continued when starting insulin, and the additional data does not contradict the reference. There is no indication that the LLM fabricated information; rather, it elaborated on the concept with plausible study results. Therefore, the LLM's response is accurate and not hallucinated.

        Judgment: not hallucinated  
        Score: 90"""
        # Extract the score using regex
        match = re.search(r"Score:\s*(\d+)", check_hallucination_response)
        score = int(match.group(1)) if match else 49 #YQ: 49 is the default score.
        return score
    
    # === Test the RAG system ===
    def test_rag(self, indices):
        keys = [list(self.dataset.data.keys())[idx] for idx in indices]
        bm25 = self.dataset.bm25
        bm25_corpus = self.dataset.bm25_corpus
        
        results = {}
        
        for key in tqdm(keys):  # test on small subset
            question = self.dataset.data[key]["question"]
            gt_answer = self.dataset.data[key]["answer"]
            gt_context = self.dataset.data[key]["context"]
            print("\n" + "="*100)
            print(f"\nQuestion: {question}\n")
            
            no_rag_resp = self.no_rag_answer(question)
            no_rag_reward = self.compute_reward(question, no_rag_resp, gt_answer)
            print(f"Hallucination (No-RAG): {no_rag_reward}")
            print("-"*50)
            
            rag_resp = self.bm25_answer(question, bm25, bm25_corpus)
            rag_reward = self.compute_reward(question, rag_resp, gt_answer)
            print(f"Hallucination (RAG): {rag_reward}")
            print("-"*50)
            
            gt_resp = self.gt_answer(question, gt_context)
            gt_reward = self.compute_reward(question, gt_resp, gt_answer)
            print(f"Hallucination (GT): {gt_reward}")
            
            results[key] = {
                "no_rag_reward": no_rag_reward,
                "rag_reward": rag_reward,
                "gt_reward": gt_reward,
            }
        
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    pubmedqa = PubmedqaDataset()
    rag_pubmedqa = CompareRag(dataset=pubmedqa)
    rag_pubmedqa.test_rag()
