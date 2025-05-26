"""
This file implements various reinforcement learning algorithms for the retriever system.
The main components are:
1. Algorithm initialization and factory function
2. Base ERM (Empirical Risk Minimization) class
3. Various algorithm implementations (DRNN, ARM_CML, ARM_LL, etc.)
4. Helper classes for loss computation and metrics tracking
"""

import math
import numpy as np
import torch
from torch import nn
import higher
from model import *
from sklearn.metrics.pairwise import rbf_kernel
import os
os.environ['CURL_CA_BUNDLE'] = ''

def init_algorithm(args):
    """
    Factory function to initialize the appropriate algorithm based on command line arguments.
    
    Args:
        args: Command line arguments containing algorithm settings
        
    Returns:
        algorithm: Initialized algorithm instance
        
    The function handles:
    1. Device setup (CPU/GPU)
    2. Model architecture selection
    3. Loss function configuration
    4. Algorithm-specific hyperparameters
    """
    # Set number of training domains (usually 1 unless specified)
    num_train_domains = 1

    # Setup device (GPU if available, else CPU)
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # Initialize context network for ARM-CML algorithm
    if args.algorithm == 'ARM-CML':
        context_net = context_network()
        context_net.to(device)

    # Determine if features should be returned based on algorithm type
    return_features = args.algorithm in ['DANN', 'MMD']

    # Initialize the main policy network
    if args.algorithm == 'ARM-CML':
        model = policy_network(model_config=args.model_config,
                           add_linear=True,
                           embedding_size=args.embedding_size,
                           freeze_encoder=True, 
                           context_net=True)
    else:
        model = policy_network(model_config=args.model_config,
                           add_linear=True,
                           embedding_size=args.embedding_size,
                           freeze_encoder=True)

    model = model.to(device)

    # Initialize loss function
    if args.algorithm in ['DRNN']:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Setup algorithm-specific hyperparameters
    hparams = {}
    hparams['support_size'] = args.batch_size // args.meta_batch_size

    # Initialize the appropriate algorithm based on args.algorithm
    if args.algorithm == 'ERM':
        hparams['batch_size'] = args.batch_size
        algorithm = ERM(model, loss_fn, device, 'adam', args.lr, hparams=hparams)
    
    elif args.algorithm == 'DRNN':
        hparams['robust_step_size'] = 0.01
        algorithm = DRNN(model, loss_fn, device, 'adam', args.lr, num_train_domains, hparams)
    
    elif args.algorithm == 'DANN':
        hparams['d_steps_per_g_step'] = args.d_steps_per_g_step
        hparams['lambd'] = args.lambd
        algorithm = DANN(model, loss_fn, device,'adam', args.lr, hparams, num_train_domains, num_classes)
    
    elif args.algorithm == 'MMD':
        hparams['gamma'] = args.gamma
        algorithm = MMD(model, loss_fn, device,'adam', args.lr, hparams)
    
    elif args.algorithm == 'ARM-CML':
        hparams['batch_size'] = args.batch_size
        hparams['adapt_bn'] = args.adapt_bn
        algorithm = ARM_CML(model, loss_fn, device,'adam', args.lr, context_net, hparams)
    
    elif args.algorithm == 'ARM-LL':
        hparams['batch_size'] = args.batch_size
        hparams['gamma'] = args.gamma
        learned_loss_net = MLP(in_size=128, norm_reduce=True).to(device)
        algorithm = ARM_LL(model, loss_fn, device, 'adam', args.lr, learned_loss_net, hparams)
    
    elif args.algorithm == 'ARM-BN':
        algorithm = ARM_BN(model, loss_fn, device,'adam', args.lr, hparams)

    return algorithm

class AverageMeter:
    """
    Computes and stores the average and current value of metrics.
    Useful for tracking training progress.
    """
    def __init__(self, name):
        self.value = 0
        self.total_count = 0

    def update(self, value, count):
        """Update the meter with new value and count"""
        old_count = self.total_count
        new_count = new_count + count
        self.value = self.value * old_count / new_count + value * count / new_count
        self.total_count = new_count

    def reset(self):
        """Reset the meter to initial state"""
        self.value = 0
        self.count = 0

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
        for batch_id in range(n_batch):
            start = batch_id * self.batch_size
            end = start + self.batch_size
            end = min(len(x), end)
            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)
        logits = torch.cat(logits)
        return logits

    def update(self, loss):
        """Perform one optimization step"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, logits, labels):
        """Calculate accuracy of predictions"""
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

class DRNNLossComputer:
    """
    Implements the DRNN (Distributionally Robust Neural Networks) loss computation.
    This loss aims to be robust to worst-case distributions over groups.
    
    Adapted from public DRNN repo.
    """
    def __init__(self, criterion, is_robust, n_groups, alpha=None, gamma=0.1, adj=None,
                 min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False, device='cpu'):
        self.criterion = criterion
        self.is_robust = is_robust
        self.step_size = step_size
        self.btl = btl
        self.device = device
        self.n_groups = n_groups
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups
        
        # The following parameters are not used in current implementation
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.normalize_loss = normalize_loss

    def loss(self, yhat, y, group_idx=None, is_training=False):
        """
        Compute the robust loss.
        
        Args:
            yhat: Model predictions
            y: Ground truth labels
            group_idx: Group indices
            is_training: Whether in training mode
            
        Returns:
            actual_loss: The computed loss value
        """
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        """
        Compute the robust loss using adversarial probabilities.
        
        Args:
            group_loss: Loss per group
            group_count: Number of samples per group
            
        Returns:
            robust_loss: The robust loss value
            adv_probs: Updated adversarial probabilities
        """
        adjusted_loss = group_loss
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

class DRNN(ERM):
    """
    Distributionally Robust Neural Networks implementation.
    DRNN aims to be robust to worst-case distributions over groups in the training data.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        n_groups: Number of groups in the data
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, n_groups, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr)
        self.device = device
        self.n_groups = n_groups
        self.loss_computer = DRNNLossComputer(
            loss_fn, 
            is_robust=True,
            n_groups=n_groups,
            step_size=hparams['robust_step_size'],
            device=device
        )

    def learn(self, images, labels, group_ids):
        """
        Perform one learning step with DRNN.
        
        Args:
            images: Input images
            labels: Ground truth labels
            group_ids: Group IDs for grouped training
            
        Returns:
            logits: Model predictions
            stats: Training statistics
        """
        self.train()
        logits = self.predict(images)
        loss = self.loss_computer.loss(logits, labels, group_ids.to(self.device), is_training=True)
        self.update(loss)
        stats = {'objective': loss.detach().item()}
        return logits, stats

class ARM_CML(ERM):
    """
    Adaptive Risk Minimization with Contextual Meta-Learning.
    This algorithm adapts to test-time distribution shifts using meta-learning.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        context_net: Network for learning context embeddings
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, context_net, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr, hparams=hparams)
        self.context_net = context_net
        self.support_size = hparams['support_size']
        self.adapt_bn = hparams['adapt_bn']
        
        # Initialize optimizer with both model and context network parameters
        params = list(self.model.parameters()) + list(self.context_net.parameters())
        self.init_optimizers(params)

    def predict(self, x, test=False):
        """
        Make predictions using both the main model and context network.
        
        Args:
            x: Input data
            test: Whether in test mode
            
        Returns:
            logits: Model predictions
        """
        if test:
            return self.model(x)
            
        batch_size = len(x)
        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size

        if self.adapt_bn:
            # Handle batch normalization adaptation
            out = []
            for i in range(meta_batch_size):
                x_i = x[i*support_size:(i+1)*support_size]
                context_i = self.context_net(x_i)
                context_i = context_i.mean(dim=0).expand(support_size, -1)
                bert_xi = self.model(x_i, linear_forward=False)
                x_i = torch.cat([bert_xi, context_i], dim=1)
                out.append(self.model(x_i, bert_forward=False))
            return torch.cat(out)
        else:
            # Standard forward pass with context
            context = self.context_net(x)
            l = context.shape[1]
            context = context.reshape((meta_batch_size, support_size, l))
            context = context.mean(dim=1)
            context = torch.repeat_interleave(context, repeats=support_size, dim=0)
            bert_x = self.model(x, linear_forward=False)
            x = torch.cat([bert_x, context], dim=1)
            return self.model(x, bert_forward=False)

class ARM_LL(ERM):
    """
    Adaptive Risk Minimization with Learned Loss.
    This algorithm learns a loss function that adapts to distribution shifts.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        learned_loss_net: Network for learning the loss function
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, learned_loss_net, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr, hparams=hparams)
        self.gamma = hparams['gamma']
        self.support_size = hparams['support_size']
        self.batch_size = hparams['batch_size']
        self.learned_loss_net = learned_loss_net
        self.n_inner_iter = 3
        self.inner_lr = 1e-4
        self.inner_opt = torch.optim.Adam(model.parameters(), lr=self.inner_lr)
        
        # Initialize optimizer with both model and learned loss network parameters
        params = list(self.model.parameters()) + list(self.learned_loss_net.parameters())
        self.init_optimizers(params)

    def my_cdist(cls, x1, x2):
        """Compute custom distance between two tensors"""
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                         x1,
                         x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(cls, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        """Compute Gaussian kernel between two tensors"""
        D = cls.my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def mmd(cls, x, y, kernel='gaussian'):
        """Compute Maximum Mean Discrepancy between two distributions"""
        if kernel == 'gaussian':
            Kxx = cls.gaussian_kernel(x, x).mean()
            Kyy = cls.gaussian_kernel(y, y).mean()
            Kxy = cls.gaussian_kernel(x, y).mean()
        elif kernel == 'rbf':
            Kxx = rbf_kernel(x, x).mean()
            Kyy = rbf_kernel(y, y).mean()
            Kxy = rbf_kernel(x, y).mean()
        else:
            raise Exception("The kernel type does not exist!")
        return Kxx + Kyy - 2 * Kxy

    def predict(self, x, labels=None, backprop_loss=False, test=False):
        """
        Make predictions using the learned loss network.
        
        Args:
            x: Input data
            labels: Ground truth labels (optional)
            backprop_loss: Whether to backpropagate the loss
            test: Whether in test mode
            
        Returns:
            logits: Model predictions
            objective/loss: Loss value if backprop_loss is True
        """
        self.train()
        n_domains = 1
        n_batch = math.ceil(len(x) / self.batch_size)
        logits = []
        loss = []
        penalty = 0
        
        if test:
            return self.model(x)
            
        num_sample = 0
        for batch_id in range(n_batch):
            start = batch_id*self.batch_size
            end = start + self.batch_size
            end = min(len(x), end)
            domain_x = x[start:end]
            num_sample += len(domain_x)

            with higher.innerloop_ctx(
                self.model, self.inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Inner loop optimization
                for _ in range(self.n_inner_iter):
                    spt_logits = fnet(domain_x)
                    spt_loss = self.learned_loss_net(spt_logits)
                    diffopt.step(spt_loss)

                # Evaluate
                domain_logits = fnet(domain_x)
                logits.append(domain_logits)

                if backprop_loss and labels is not None:
                    domain_labels = labels[start:end]
                    domain_loss = self.loss_fn(domain_logits, domain_labels)
                    domain_loss.backward()
                    loss.append(domain_loss.to('cpu').detach().item())

        if self.gamma != 0:
            # Compute MMD penalty
            logits_all = self.model(x)
            for domain_id in range(n_domains):
                for j in range(domain_id, n_domains):
                    penalty_add = self.mmd(
                        logits_all[domain_id * self.support_size:(domain_id + 1) * self.support_size],
                        logits_all[j * self.support_size:(j + 1) * self.support_size]
                    )
                    penalty -= penalty_add

        logits = torch.cat(logits)
        if self.gamma != 0:
            penalty /= (n_domains * (n_domains - 1) / 2)
            objective = self.gamma * penalty
            return logits, objective
        else:
            if backprop_loss:
                return logits, np.mean(loss)
            else:
                return logits

    def learn(self, x, labels, group_ids=None):
        """
        Perform one learning step.
        
        Args:
            x: Input data
            labels: Ground truth labels
            group_ids: Optional group IDs
            
        Returns:
            logits: Model predictions
            stats: Training statistics
        """
        self.train()
        logits, loss = self.predict(x, labels, backprop_loss=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return logits, {}

class ARM_BN(ERM):
    """
    Adaptive Risk Minimization with Batch Normalization.
    This algorithm adapts batch normalization statistics at test time.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr)
        self.support_size = hparams['support_size']

    def predict(self, x):
        """
        Make predictions with adapted batch normalization.
        
        Args:
            x: Input data
            
        Returns:
            logits: Model predictions
        """
        self.model.train()
        n_domains = math.ceil(len(x) / self.support_size)
        logits = []
        
        for domain_id in range(n_domains):
            start = domain_id * self.support_size
            end = start + self.support_size
            end = min(len(x), end)
            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)

        logits = torch.cat(logits)
        return logits

class MMD(ERM):
    """
    Maximum Mean Discrepancy implementation.
    This algorithm minimizes the MMD distance between source and target distributions.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        optimizer_name: Name of optimizer to use
        lr: Learning rate
        hparams: Additional hyperparameters
    """
    def __init__(self, model, loss_fn, device, optimizer_name, lr, hparams):
        super().__init__(model, loss_fn, device, optimizer_name, lr)
        self.gamma = hparams['gamma']
        self.support_size = hparams['support_size']

    def my_cdist(cls, x1, x2):
        """Compute custom distance between two tensors"""
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                         x1,
                         x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(cls, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        """Compute Gaussian kernel between two tensors"""
        D = cls.my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def mmd(cls, x, y, kernel='gaussian'):
        """Compute Maximum Mean Discrepancy between two distributions"""
        if kernel == 'gaussian':
            Kxx = cls.gaussian_kernel(x, x).mean()
            Kyy = cls.gaussian_kernel(y, y).mean()
            Kxy = cls.gaussian_kernel(x, y).mean()
        elif kernel == 'rbf':
            Kxx = rbf_kernel(x, x).mean()
            Kyy = rbf_kernel(y, y).mean()
            Kxy = rbf_kernel(x, y).mean()
        else:
            raise Exception("The kernel type does not exist!")
        return Kxx + Kyy - 2 * Kxy

class DANN(ERM):
    """
    Domain Adversarial Neural Network implementation.
    This algorithm learns domain-invariant features through adversarial training.
    
    Args:
        model: The neural network model
        loss_fn: Loss function to optimize
        device: Device to run computations on
        hparams: Additional hyperparameters
        num_domains: Number of domains
        num_classes: Number of classes
    """
    def __init__(self, model, loss_fn, device, hparams, num_domains, num_classes):
        # Setup feature extractor and classifier
        featurizer = model
        classifier = nn.Linear(model.num_features, num_classes)
        model = nn.Sequential(featurizer, classifier).to(device)
        
        super().__init__(model, loss_fn, device, hparams, init_optim=False)
        
        # Setup domain discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(featurizer.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains),
        ).to(device)

        self.num_domains = num_domains
        self.featurizer = featurizer.to(device)
        self.classifier = classifier.to(device)
        self.support_size = hparams['support_size']
        self.d_steps_per_g_step = hparams['d_steps_per_g_step']
        self.lambd = hparams['lambd']
        self.step_count = 0
        
        self.init_optimizers(None)

    def init_optimizers(self, params):
        """Initialize optimizers for generator and discriminator"""
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
            self.disc_optimizer = torch.optim.SGD(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )

    def learn(self, images, labels, group_ids):
        """
        Perform one learning step with adversarial training.
        
        Args:
            images: Input images
            labels: Ground truth labels
            group_ids: Domain/group IDs
            
        Returns:
            logits: Model predictions
            stats: Training statistics
        """
        self.train()
        
        # Extract features
        features = self.featurizer(images)
        
        # Discriminator predictions
        domain_preds = self.discriminator(features)
        group_ids = group_ids.to(self.device)
        disc_loss = self.loss_fn(domain_preds, group_ids)
        
        # Alternate between training discriminator and generator
        if self.step_count % (self.d_steps_per_g_step + 1) < self.d_steps_per_g_step:
            # Train discriminator
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
            self.step_count += 1
            return None, None
        else:
            # Train generator
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)
            gen_loss = loss - self.lambd * disc_loss
            self.optimizer.zero_grad()
            gen_loss.backward()
            self.optimizer.step()
            self.step_count += 1
            return logits, {}

# Additional algorithm implementations would follow here...
# Each would have similar detailed documentation explaining their purpose and functionality 