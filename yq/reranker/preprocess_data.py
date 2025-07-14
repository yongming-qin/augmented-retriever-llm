"""
for all combinations of question and context, compute the reward.
save the reward to a json file.


"""
import os
import json
import time
import threading
from tqdm import tqdm
from rag_dataset import PubmedqaDataset, CompareRag
from concurrent.futures import ThreadPoolExecutor, as_completed

class PreprocessData:
    def __init__(self, dataset: PubmedqaDataset, compare_rag: CompareRag):
        self.dataset = dataset
        self.compare_rag = compare_rag
        self.cache_path = "reward_cache.json"
        
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        
        if os.path.exists(self.cache_path):
            self.load_cache()
        else:
            self.reward_cache = {}
            
    def save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.reward_cache, f)
            print(f"Saved {len(self.reward_cache)} rewards to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_cache(self):
        with open(self.cache_path, "r") as f:
            self.reward_cache = json.load(f)
        print(f"Loaded {len(self.reward_cache)} rewards from cache")
        
    def _cache_key(self, question, context):
        return f"{question} ||| {context}"
        
    def preprocess(self):
        bm25 = self.dataset.bm25
        bm25_corpus = self.dataset.bm25_corpus
        for key in tqdm(self.dataset.data.keys()):
            question = self.dataset.data[key]["question"]
            answer = self.dataset.data[key]["answer"]
            gt_context = self.dataset.data[key]["context"]
            
            retrieved_ctxs = bm25.get_top_n(question.split(), bm25_corpus, n=3)
            ##YQ: each retrieved context is just a string, ensured by the code.
            contexts = ["NO CONTEXT IS NEEDED", gt_context] + retrieved_ctxs
            
            for ctx in contexts:
                key = self._cache_key(question, ctx)
                if key not in self.reward_cache:
                    llm_answer = self.compare_rag.gt_answer(question, ctx)
                    reward = self.compare_rag.compute_reward(question, llm_answer, answer)
                    self.reward_cache[key] = reward
                    
    def _rate_limited_call(self, func, *args, **kwargs):
        with self.rate_limit_lock:
            elapsed = time.time() - self.last_request_time
            wait_time = max(0, 1.0 - elapsed)  # 1 QPS = 1 second between calls
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            self.last_request_time = time.time()
            return result
        
    def parallel_preprocess(self, num_workers=4):
        bm25 = self.dataset.bm25
        bm25_corpus = self.dataset.bm25_corpus
        queries_cached = []
        queries_not_cached = []
        for key in self.dataset.data.keys():
            question = self.dataset.data[key]["question"]
            gt_answer = self.dataset.data[key]["answer"]
            gt_context = self.dataset.data[key]["context"]
            retrieved_ctxs = bm25.get_top_n(question.split(), bm25_corpus, n=3)
            ##YQ: each retrieved context is just a string, ensured by the code.
            contexts = ["NO CONTEXT IS NEEDED", gt_context] + retrieved_ctxs
            
            for ctx in contexts:
                key = self._cache_key(question, ctx)
                if key in self.reward_cache:
                    queries_cached.append((question, ctx, gt_answer))
                else:
                    queries_not_cached.append((question, ctx, gt_answer))
        
        def preprocess_query(question, ctx, gt_answer):
            llm_answer = self.compare_rag.gt_answer(question, ctx)
            reward = self.compare_rag.compute_reward(question, llm_answer, gt_answer)
            return self._cache_key(question, ctx), reward
    
        # parallel preprocess the queries_not_cached
        completed_queries = 0
        with tqdm(total=len(queries_not_cached) + len(queries_cached), initial=len(queries_cached), desc=f"Preprocessing queries in parallel with {num_workers} workers") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._rate_limited_call, preprocess_query, query[0], query[1], query[2]) for query in queries_not_cached]
                for future in as_completed(futures):
                    key, reward = future.result()
                    self.reward_cache[key] = reward
                    completed_queries += 1
                    if completed_queries % 100 == 0:
                        self.save_cache()
                    pbar.update(1)
        
if __name__ == "__main__":
    dataset = PubmedqaDataset()
    compare_rag = CompareRag(dataset)
    preprocess_data = PreprocessData(dataset, compare_rag)
    try:
        # preprocess_data.preprocess()
        preprocess_data.parallel_preprocess()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving cache...")
    finally:
        preprocess_data.save_cache()
    