from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''
class policy_network(nn.Module):

    def __init__(self,
                 model_config="bert-base-uncased",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True,
                 context_net=False) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained(model_config)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            if context_net:
                input_dim = self.model.config.hidden_size * 2
            else:
                input_dim = self.model.config.hidden_size
            self.linear = nn.Linear(input_dim,
                                    embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list, bert_forward=True, linear_forward=True):
        if bert_forward:
            input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
            # print(f"input: {input}")
            output = self.model(**input, output_hidden_states=True)
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]
            # Get [CLS] hidden states
            sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
            # print(f"sentence_embedding: {sentence_embedding}")

        if linear_forward:
            if self.linear:
                if bert_forward:
                    sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size
                else:
                    sentence_embedding = self.linear(input_list)
        return sentence_embedding


class context_network(nn.Module):

    def __init__(self,
                 model_config='all-MiniLM-L6-v2',
                 add_linear=True,
                 embedding_size=768,
                 freeze_encoder=True) -> None:
        super().__init__()
        print("context model_config:", model_config)
        self.model = SentenceTransformer(model_config)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(384, embedding_size)  #embedding_size is 768 for bert-base-uncased,
            # distilbert-base-uncased, output dimension of all-MiniLM-L6-v2 is 384
        else:
            self.linear = None

    def forward(self, input_list):
        sentence_embedding = self.model.encode(input_list, convert_to_tensor=True)
        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


def test_policy_network():
    test_pids = [1]
    cand_pids = [0, 2, 4]
    problems = [
        "This is problem 0", "This is the first question", "Second problem is here", "Another problem",
        "This is the last problem"
    ]
    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]

    model = policy_network(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
    scores = model(ctxt_list, cands_list).cpu().detach().numpy()
    print(f"scores: {scores}")
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        scores = scores[i, :].tolist()
        cand_rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")

class MLP(nn.Module):
    def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
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
        out = self.model(x)
        if self.norm_reduce:
            out = torch.norm(out)

        return out



if __name__ == "__main__":
    test_policy_network()
