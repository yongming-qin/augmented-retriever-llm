"""
first, I want to test whether the common embeddings like open ai or hugging face is time/version-aware.
Please write some code that I can have a try on these common embedding models.
If I need some rag database of different versions of code api documentation or different time's documentation.
Please suggest where to download them and write the code to use them in the RAG.

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
    "mpnet": "all-mpnet-base-v2"
}

openai_models = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072  # optional, slower + more expensive
}

# Versioned inputs and related queries
inputs = [
    "[VERSION:1.0] how to add one to a number using foo function ",
    "[VERSION:2.0] how to multiply two to a number using foo function ",
    "[VERSION:3.0] how to subtract one from a number using foo function ",
    "[VERSION:1.0] def foo(x): return x + 1",
    "[VERSION:2.1] def foo(x): return x * 2",
    "[VERSION:3.0] def foo(x): return x - 1",
    "[VERSION:1.2] def boo(x): return x + 10",
    "[VERSION:0.1] def boo(x): return x * 20",
    "[VERSION:0.3] def boo(x): return x - 10",
]

output_file = "embedding_similarities_version_aware.txt"

# Helper: calculate and log cosine similarity matrix
def log_cosine_similarities(model_name, embeddings, inputs, f):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_scores = cosine_similarity(embeddings)
    f.write(f"\n=== Model: {model_name} ===\n")
    for i in range(0,3):
        f.write(f"\nQuery: {inputs[i]}\n")
        for j in range(3,len(inputs)):
            f.write(f"  Similarity to '{inputs[j]}': {cosine_scores[i][j]:.3f}\n")

with open(output_file, "w") as f:
    # Hugging Face models
    for name, path in hf_models.items():
        print(f"Processing HF model: {name}")
        model = SentenceTransformer(path)
        emb = model.encode(inputs, convert_to_numpy=True, show_progress_bar=False)
        log_cosine_similarities(name, emb, inputs, f)

    # OpenAI models
    for name, dim in openai_models.items():
        print(f"Processing OpenAI model: {name}")
        embeddings = []
        for text in inputs:
            try:
                resp = openai.embeddings.create(model=name, input=text)
                embeddings.append(resp.data[0].embedding)
                time.sleep(0.5)  # Avoid hitting rate limits
            except Exception as e:
                print(f"OpenAI error: {e}")
                embeddings.append([0] * dim)
        embeddings = np.array(embeddings)
        log_cosine_similarities(f"OpenAI-{name}", embeddings, inputs, f)

print(f"\nResults saved to: {output_file}")

