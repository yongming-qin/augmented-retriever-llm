"""
This module implements the neural network architectures used in the retriever system.
The main components are:

1. policy_network: A BERT-based model for encoding text and making predictions
2. context_network: A sentence transformer model for encoding contextual information
3. MLP: A simple multi-layer perceptron for auxiliary tasks
4. Testing utilities for the policy network

Key Features:
- Uses pre-trained transformer models (BERT and Sentence Transformers)
- Supports freezing encoder weights for transfer learning
- Includes optional linear projection layers for dimensionality reduction
- Handles both regular forward passes and context-aware processing
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''

class policy_network(nn.Module):
    """
    A neural network policy that uses BERT for text encoding.
    
    This network can:
    1. Encode text using a pre-trained BERT model
    2. Optionally freeze the BERT encoder
    3. Add an optional linear projection layer
    4. Handle context-aware processing when needed
    
    Architecture:
    - BERT encoder (frozen or trainable)
    - Optional linear projection layer
    - Support for context concatenation
    
    Args:
        model_config (str): Name of the pre-trained BERT model to use (default: "bert-base-uncased")
        add_linear (bool): Whether to add a linear projection layer after BERT (default: False)
        embedding_size (int): Size of the final embeddings if using linear projection (default: 128)
        freeze_encoder (bool): Whether to freeze the BERT encoder weights (default: True)
        context_net (bool): Whether this model will receive context embeddings (default: False)
    """
    def __init__(self,
                 model_config="bert-base-uncased",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True,
                 context_net=False) -> None:
        super().__init__()
        
        # Initialize BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained(model_config)

        # Optionally freeze the BERT encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add optional linear projection layer
        if add_linear:
            self.embedding_size = embedding_size
            # Double input dimension if using context network (concatenating context embeddings)
            if context_net:
                input_dim = self.model.config.hidden_size * 2
            else:
                input_dim = self.model.config.hidden_size
            # Create linear layer (BERT hidden size -> embedding_size)
            self.linear = nn.Linear(input_dim, embedding_size)
        else:
            self.linear = None

    def forward(self, input_list, bert_forward=True, linear_forward=True):
        """
        Forward pass through the network.
        
        The forward pass can be customized to:
        1. Only use BERT encoding
        2. Only use linear projection
        3. Use both BERT encoding and linear projection
        
        Args:
            input_list: List of input texts to process
            bert_forward (bool): Whether to run BERT encoding (default: True)
            linear_forward (bool): Whether to run linear projection (default: True)
            
        Returns:
            sentence_embedding: Final embeddings for the input texts
        """
        if bert_forward:
            # Tokenize and encode with BERT
            input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
            output = self.model(**input, output_hidden_states=True)
            
            # Extract [CLS] token embedding from last layer
            last_hidden_states = output.hidden_states[-1]
            sentence_embedding = last_hidden_states[:, 0, :]  # [batch_size, hidden_size]

        # Apply optional linear projection
        if linear_forward and self.linear:
            if bert_forward:
                sentence_embedding = self.linear(sentence_embedding)  # [batch_size, embedding_size]
            else:
                sentence_embedding = self.linear(input_list)
                
        return sentence_embedding


class context_network(nn.Module):
    """
    A network for encoding contextual information using Sentence Transformers.
    
    This network:
    1. Uses a pre-trained Sentence Transformer model
    2. Can freeze the encoder weights
    3. Adds an optional linear projection layer
    
    Architecture:
    - Sentence Transformer encoder
    - Optional linear projection layer
    
    Args:
        model_config (str): Name of the Sentence Transformer model (default: 'all-MiniLM-L6-v2')
        add_linear (bool): Whether to add a linear projection layer (default: True)
        embedding_size (int): Size of the final embeddings if using linear projection (default: 768)
        freeze_encoder (bool): Whether to freeze the encoder weights (default: True)
    """
    def __init__(self,
                 model_config='all-MiniLM-L6-v2',
                 add_linear=True,
                 embedding_size=768,
                 freeze_encoder=True) -> None:
        super().__init__()
        print("context model_config:", model_config)
        self.model = SentenceTransformer(model_config)

        # Optionally freeze the encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add optional linear projection layer
        if add_linear:
            self.embedding_size = embedding_size
            # Project from Sentence Transformer output size (384) to desired embedding size
            self.linear = nn.Linear(384, embedding_size)
        else:
            self.linear = None

    def forward(self, input_list):
        """
        Forward pass through the context network.
        
        Args:
            input_list: List of input texts to encode
            
        Returns:
            sentence_embedding: Context embeddings for the input texts
        """
        # Encode texts using Sentence Transformer
        sentence_embedding = self.model.encode(input_list, convert_to_tensor=True)
        
        # Apply optional linear projection
        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # [batch_size, embedding_size]
            
        return sentence_embedding


class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) network.
    
    Architecture:
    - 4 linear layers with ReLU activation
    - Optional L2 norm reduction of output
    
    Args:
        in_size (int): Input dimension (default: 10)
        out_size (int): Output dimension (default: 1)
        hidden_dim (int): Hidden layer dimension (default: 32)
        norm_reduce (bool): Whether to apply L2 norm to output (default: False)
    """
    def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
        
        # Build 4-layer MLP with ReLU activation
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_size),
        )
        
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            out: Output tensor, optionally L2 normalized
        """
        out = self.model(x)
        if self.norm_reduce:
            out = torch.norm(out)
        return out


def test_policy_network():
    """
    Test function for the policy network.
    
    This function:
    1. Creates sample problems and candidate solutions
    2. Initializes a policy network
    3. Computes similarity scores between problems and candidates
    4. Ranks candidates based on scores
    
    The test helps verify that the policy network can:
    - Process text inputs correctly
    - Generate meaningful embeddings
    - Compute reasonable similarity scores
    """
    # Setup test data
    test_pids = [1]  # Test problem IDs
    cand_pids = [0, 2, 4]  # Candidate problem IDs
    problems = [
        "This is problem 0",
        "This is the first question",
        "Second problem is here",
        "Another problem",
        "This is the last problem"
    ]
    
    # Get test and candidate texts
    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]

    # Initialize and run model
    model = policy_network(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
    scores = model(ctxt_list, cands_list).cpu().detach().numpy()
    
    # Print results
    print(f"scores: {scores}, {scores.shape=}")
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        scores = scores[i, :].tolist()
        # Rank candidates by score
        cand_rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")


if __name__ == "__main__":
    test_policy_network() 