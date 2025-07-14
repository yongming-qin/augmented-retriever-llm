"""
Test time-aware embedding models.
Yongming Qin
2025-06-17
"""


from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np
import time
from dotenv import load_dotenv

load_dotenv()

# Define models to test
hf_models = {
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "miniLM": "all-MiniLM-L6-v2",
    # "mpnet": "all-mpnet-base-v2"
}

openai_models = {
    "text-embedding-3-small": 1536,
    # "text-embedding-3-large": 3072
}

# Time-stamped queries and documents simulating knowledge at different points
inputs = [
    "[DATE:2020-11-01] how to use foo function",
    "[DATE:2022-02-01] how to use boo function",
    "[DATE:2025-01-01] how to increment a number using foo function",
    # "[DATE:2022-01-01] how to double a number using foo function",
    # "[DATE:2024-01-01] how to subtract one from a number using foo function",
    "[DATE:2020-01-01] def foo(x): return x + 1",
    "[DATE:2022-01-01] def foo(x): return x * 2",
    "[DATE:2024-01-01] def foo(x): return x - 1",
    "[DATE:2020-01-01] def boo(x): return x + 10",
    "[DATE:2022-01-01] def boo(x): return x * 20",
    "[DATE:2024-01-01] def boo(x): return x - 10",
]

output_file = "embedding_similarities_time_aware.txt"

# Helper to compute and log similarities between queries and documents
def log_cosine_similarities(model_name, embeddings, inputs, f):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_scores = cosine_similarity(embeddings)
    f.write(f"\n=== Model: {model_name} ===\n")
    for i in range(3):  # first 3 are queries
        f.write(f"\nQuery: {inputs[i]}\n")
        for j in range(3, len(inputs)):
            f.write(f"  Similarity to '{inputs[j]}': {cosine_scores[i][j]:.3f}\n")

with open(output_file, "w") as f:
    # Hugging Face embeddings
    for name, path in hf_models.items():
        print(f"Processing Hugging Face model: {name}")
        model = SentenceTransformer(path)
        emb = model.encode(inputs, convert_to_numpy=True, show_progress_bar=False)
        log_cosine_similarities(name, emb, inputs, f)

    # OpenAI embeddings
    for name, dim in openai_models.items():
        print(f"Processing OpenAI model: {name}")
        embeddings = []
        for text in inputs:
            try:
                resp = openai.embeddings.create(model=name, input=text)
                embeddings.append(resp.data[0].embedding)
                time.sleep(0.5)
            except Exception as e:
                print(f"OpenAI error: {e}")
                embeddings.append([0] * dim)
        embeddings = np.array(embeddings)
        log_cosine_similarities(f"OpenAI-{name}", embeddings, inputs, f)

print(f"\nResults saved to: {output_file}")
