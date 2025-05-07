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

    # if 'MATH' in args.data_root_test[0]:
    #     num_train_domains = 7
    #
    # else:
    num_train_domains = 1

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    # Channels of main model
    if args.algorithm == 'ARM-CML':
        context_net = context_network()
        context_net.to(device)


    if args.algorithm in ['DANN', 'MMD']:
        return_features = True
    else:
        return_features = False

    # Main model
    ## policy network
    if args.algorithm == 'ARM-CML':
        model = policy_network(model_config=args.model_config,
                               add_linear=True,
                               embedding_size=args.embedding_size,
                               freeze_encoder=True, context_net=True)
    else:
        model = policy_network(model_config=args.model_config,
                                      add_linear=True,
                                      embedding_size=args.embedding_size,
                                      freeze_encoder=True)


    model = model.to(device)


    # Loss fn
    if args.algorithm in ['DRNN']:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Algorithm
    hparams = {}
    hparams['support_size'] = args.batch_size // args.meta_batch_size
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

    def __init__(name):
        self.value = 0
        self.total_count = 0

    def update(self, value, count):
        old_count = self.total_count
        new_count = new_count + count

        self.value = self.value * old_count / new_count + value * count / new_count
        self.total_count = new_count

    def reset(self):
        self.value = 0
        self.count = 0

class ERM(nn.Module):
    def __init__(self, model, loss_fn, device, optimizer_name, lr, init_optim=True,hparams={}, **kwargs):
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
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, params),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay)

    def predict(self, x, test=False):
        n_batch = math.ceil(len(x) / self.batch_size)
        logits = []
        for batch_id in range(n_batch):
            start = batch_id * self.batch_size
            end = start + self.batch_size
            end = min(len(x), end)  # in case final domain has fewer than support size samples

            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)
        logits = torch.cat(logits)
        return logits

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, logits, labels):
        # Evaluate
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
        return accuracy

    def learn(self, images, labels, group_ids=None):

        self.train()

        # Forward
        logits = self.predict(images)
        loss = self.loss_fn(logits, labels)


        self.update(loss)

        stats = {
                 'objective': loss.detach().item(),
                }

        return logits, stats

class DRNNLossComputer:
    """ Adapted from public DRNN repo"""
    def __init__(self, criterion, is_robust, n_groups, alpha=None, gamma=0.1, adj=None,
                 min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False, device='cpu'):

        self.criterion = criterion
        self.is_robust = is_robust
        self.step_size = step_size

        self.btl = btl # set to false
        self.device = device

        self.n_groups = n_groups

        # quantities mintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups

        # The following 4 variables are not used
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.normalize_loss = normalize_loss

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count) # this one is actually used
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss

        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count


class DRNN(ERM):

    def __init__(self, model, loss_fn, device, optimizer_name, lr, n_groups,  hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr)

        self.device = device
        self.n_groups = n_groups
        self.loss_computer = DRNNLossComputer(loss_fn, is_robust=True,
                                         n_groups=n_groups,
                                         step_size=hparams['robust_step_size'],
                                         device=device)

    def learn(self, images, labels, group_ids):
        self.train()

        # Forward and backward
        logits = self.predict(images)
        loss = self.loss_computer.loss(logits, labels, group_ids.to(self.device), is_training=True)

        self.update(loss)

        stats = {
                'objective': loss.detach().item()
                }

        return logits, stats

class ARM_CML(ERM):

    def __init__(self, model, loss_fn, device, optimizer_name, lr, context_net, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr, hparams=hparams)

        self.context_net = context_net
        self.support_size = hparams['support_size']
        self.adapt_bn = hparams['adapt_bn']

        params = list(self.model.parameters()) + list(self.context_net.parameters())
        self.init_optimizers(params)


    def predict(self, x, test=False):
        if test:
            return self.model(x)
        batch_size = len(x)

        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size

        if self.adapt_bn:
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
            context = self.context_net(x)
            l = context.shape[1]
            context = context.reshape((meta_batch_size, support_size, l))
            context = context.mean(dim=1) # Shape: meta_batch_size, self.n_context_channels
            context = torch.repeat_interleave(context, repeats=support_size, dim=0) # meta_batch_size * support_size, context_size
            bert_x = self.model(x, linear_forward=False)
            x = torch.cat([bert_x, context], dim=1)
            return self.model(x, bert_forward=False)

class ARM_LL(ERM):

    def __init__(self, model, loss_fn, device, optimizer_name, lr, learned_loss_net, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr, hparams=hparams)

        self.gamma = hparams['gamma']
        self.support_size = hparams['support_size']
        self.batch_size = hparams['batch_size']
        self.learned_loss_net = learned_loss_net
        #self.n_inner_iter = 1, 3 #1 for tabmwp and theory #3 for algebra
        self.n_inner_iter = 3
        #self.inner_lr = 1e-1  1e-4 for tabmwp 1e-2 for theory and algebra
        self.inner_lr = 1e-4
        #self.inner_opt = torch.optim.SGD(model.parameters(),
        #                                lr=self.inner_lr)
        self.inner_opt = torch.optim.Adam(model.parameters(),
                                         lr=self.inner_lr)

        params = list(self.model.parameters()) + list(self.learned_loss_net.parameters())
        self.init_optimizers(params)

    def my_cdist(cls, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(cls, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                          1000]):
        D = cls.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(cls, x, y, kernel='gaussian'):
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
        #self.inner_opt = torch.optim.SGD(self.model.parameters(),
        #                                lr=self.inner_lr)

        self.train() # see this thread for why this is done https://github.com/facebookresearch/higher/issues/55

        #n_domains = math.ceil(len(x) / self.support_size)
        n_domains = 1
        n_batch = math.ceil(len(x) / self.batch_size)
        #print("num bach:", n_batch)
        logits = []
        loss = []
        penalty = 0
        if test:
            return self.model(x)
        num_sample = 0
        for batch_id in range(n_batch):
            start = batch_id*self.batch_size
            end = start + self.batch_size
            end = min(len(x), end) # in case final domain has fewer than support size samples

            domain_x = x[start:end]
            num_sample += len(domain_x)

            with higher.innerloop_ctx(
                self.model, self.inner_opt, copy_initial_weights=False) as (fnet, diffopt):

                # Inner loop
                for _ in range(self.n_inner_iter):
                    spt_logits = fnet(domain_x)
                    spt_loss = self.learned_loss_net(spt_logits)
                    diffopt.step(spt_loss)

                # Evaluate should be the same as promptpg
                domain_logits = fnet(domain_x)
                logits.append(domain_logits)

                if backprop_loss and labels is not None:
                    domain_labels = labels[start:end]
                    domain_loss = self.loss_fn(domain_logits, domain_labels)
                    domain_loss.backward()
                    loss.append(domain_loss.to('cpu').detach().item())

        if self.gamma != 0:
            logits_all = self.model(x)
            for domain_id in range(n_domains):
                for j in range(domain_id, n_domains):
                    penalty_add = self.mmd(logits_all[domain_id * self.support_size:(domain_id + 1) * self.support_size],
                                           logits_all[j * self.support_size:(j + 1) * self.support_size])
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
        self.train()
        logits, loss = self.predict(x, labels, backprop_loss=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = {}

        return logits, stats


class ARM_BN(ERM):

    def __init__(self, model, loss_fn, device,optimizer_name, lr, hparams={}):
        super().__init__(model, loss_fn, device, optimizer_name, lr)

        self.support_size = hparams['support_size']

    def predict(self, x):
        self.model.train()

        n_domains = math.ceil(len(x) / self.support_size)

        logits = []
        for domain_id in range(n_domains):
            start = domain_id * self.support_size
            end = start + self.support_size
            end = min(len(x), end) # in case final domain has fewer than support size samples
            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)

        logits = torch.cat(logits)

        return logits

class MMD(ERM):

    def my_cdist(cls, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(cls, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = cls.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(cls, x, y, kernel='gaussian'):
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

    def __init__(self, model, loss_fn, device, optimizer_name, lr, hparams):
        super().__init__(model, loss_fn, device, optimizer_name, lr)

        self.gamma = hparams['gamma']
        self.support_size = hparams['support_size']

    def predict(self, x, test=False):
        #self.model.train()

        # features = self.featurizer(images)
        # logits = self.classifier(features)
        # loss = self.loss_fn(logits, labels)
        features = self.model(x)
        if test:
            return features
        penalty = 0

        num_domains = math.ceil(len(x) / self.support_size) # meta batch size

        for i in range(num_domains):
            for j in range(i, num_domains):
                penalty_add = self.mmd(features[i*self.support_size:(i+1)*self.support_size], features[j*self.support_size:(j+1)*self.support_size])
                penalty -= penalty_add
        penalty /= (num_domains * (num_domains - 1) / 2)
        objective = self.gamma * penalty



        return features, objective


class DANN(ERM):
    def __init__(self, model, loss_fn, device, hparams, num_domains, num_classes):

        featurizer = model
        classifier = nn.Linear(model.num_features, num_classes)

        model = nn.Sequential(featurizer, classifier).to(device)

        super().__init__(model, loss_fn, device, hparams, init_optim=False)

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
        if self.optimizer_name == 'adam': # This is used for MNIST.
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate, weight_decay=self.weight_decay)
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay,
                                              momentum=0.9)
            self.disc_optimizer = torch.optim.SGD(self.discriminator.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay,
                                                momentum=0.9)



    def learn(self, images, labels, group_ids):
        self.train()


        # Forward
        features = self.featurizer(images)

        domain_preds = self.discriminator(features)
        group_ids = group_ids.to(self.device)
        disc_loss = self.loss_fn(domain_preds, group_ids)

        # Backprop
        if self.step_count % (self.d_steps_per_g_step + 1) < self.d_steps_per_g_step:

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            self.step_count += 1
            return None, None
        else:
            logits = self.classifier(features)

            # Evaluate
            loss = self.loss_fn(logits, labels)
            gen_loss = loss - self.lambd * disc_loss
            self.optimizer.zero_grad()
            gen_loss.backward()
            self.optimizer.step()

            stats = {}
            self.step_count += 1
            return logits, stats