"""
Practice with the algorithm_documented.py file.
Yongming, Chenxi, 2025-05-10
"""
import math
import numpy as np
import torch
import torch.nn as nn
from model import policy_network

import import_ipynb
import run_gpt3_yq_test as test



class ERM(nn.Module):
    """
    Base class implementing Empirical Risk Minimization.
    This is the simplest form of training where we directly minimize the loss on training data.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        init_optim: Whether to initialize optimizer
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, init_optim=True, hparams={}, **kwargs):
        super().__init__()
        self.batch_size = hparams["batch_size"]
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer_name = optimizer_name
        self.learning_rate = lr 

        if init_optim:
            params = self.model.parameters()
            self.init_optimizers(params)

    def init_optimizers(self, params):
        """Initialize the optimizer with the given parameters"""
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, params),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay)

    def predict(self, x, test=False):
        """
        Make predictions on input data.

        Args:
            x: Input data
            test: Whether in test mode
            
        Returns:
            logits: Model predictions
        """
        n_batch = math.ceil(len(x) / self.batch_size)
        logits = []
        self.model.eval()
        with torch.no_grad():
            for i in range(n_batch):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(x))
                batch_x = x[start_idx:end_idx]
                batch_x = batch_x.to(self.device)
                logits.append(self.model(batch_x))
        logits = torch.cat(logits)
        return logits
    
    def update(self, loss):
        """
        Perform one optimization step

        Args:
            loss: Loss value to optimize
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, logits, labels):
        """
        Calculate accuracy of predictions

        Args:
            logits: Model predictions
            labels: Ground truth labels
            
        """
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
        return accuracy

    def learn(self, images, labels, group_ids=None):
        """
        Perform one learning step.

        Args:
            images: Input images
            labels: Ground truth labels
            group_ids: Optional group IDs for grouped training
        
        Returns:
            logits: Model predictions
            stats: Training statistics
        """
        self.train()
        logits = self.predict(images)
        loss = self.loss_fn(logits, labels)
        self.update(loss)
        stats = {'objective': loss.detach().item()}
        return logits, stats


def init_algorithm(args):
    """
    Factory function to initialize the appropriate algorithm based on command line arguments.
    """
    
    device = torch.device("cuda")
    
    ## assuming args.algorithm is ERM
    model = policy_network(model_config=args.model_config,
                           add_linear=True,
                           embedding_size=args.embedding_size,
                           freeze_encoder=True)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # Setup algorithm-specific hyperparameters
    hparams = {}
    hparams['support_size'] = args.batch_size // args.meta_batch_size
    
    hparams['batch_size'] = args.batch_size
    algorithm = ERM(model, loss_fn, device, 'adam', args.lr, hparams=hparams)
    
        
    
init_algorithm(test.args)