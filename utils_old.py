import numpy as np
import os, sys
import torch
from torch import nn, optim
import subprocess

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def calc_mi(model, test_data_batch):
    # calc_mi_v3
    import math 
    from modules.utils import log_sum_exp
    
    if not model.encoder.useGaussian:
        return np.nan
    
    mi = 0
    num_examples = 0
    
    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch_data in test_data_batch:
        mu, logvar = model.encoder.input_to_posterior(batch_data)
        x_batch, dim_z = mu.size()
        ##print(x_batch, end=' ')
        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*dim_z*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy += (-0.5 * dim_z * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]

    neg_entropy = neg_entropy / num_examples
    ##print()

    num_examples = 0
    log_qz = 0.
    for i in range(len(mu_batch_list)):
        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        
        # [z_batch, 1, dim_z]
        z_samples = model.encoder.posterior_to_zs(mu, logvar, 1)
        z_samples = z_samples.view(-1, 1, dim_z)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, dim_z]
        #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, dim_z = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, dim_z)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (dim_z * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi.item()


def calc_au(model, test_data_batch, delta=0.01):
    """compute the number of active units
    """
    all_mu = []
    for batch_data in test_data_batch:
        mu, _ = model.encoder.input_to_posterior(batch_data)
        all_mu.append(mu)
    all_mu = torch.cat(all_mu, dim=0)
    mean_mu = torch.mean(all_mu, dim=0)
    var_mu = torch.var(all_mu, dim=0)
    return (var_mu >= delta).sum().item(), var_mu
