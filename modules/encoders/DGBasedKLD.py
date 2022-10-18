import torch
import numpy as np
from .utils import *
from ..utils import exp_mean_log

# implementation of Eq. 11 in paper for Gaussian distribution
class DGBasedGaussianKLD(torch.nn.Module):
    """ implementation of KL(agg-post || prior) through Mentor Carlo approximation on Density Gap
    where:
        1. agg-post = mean_i(post_i), post_i \sim N(\mu_i,\sigma_i^2)
        2. prior \sim N(0,1)
    mode: "narrow" or "wide"
        "narrow": sum of dimension-wised distances,
            e.g. for prior = [[1,2,1],
                              [2,4,2],
                              [1,2,1]]
                 post = [[0,4,0], is just ok, as marginal(prior) == marginal(post) on each dimension
                         [4,0,4],
                         [0,4,0]]
        "wide": strictly distribution-wised distance, this needs prior == post on each position in R^dim_z
    """
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return f"DGBasedGaussianKLD"
    
    def forward(self, mean, logvar, n_samples=32, agg_size=None):
        """ KL(agg-post || prior)
        KL(q||p) = \int_z [q(z)log(q(z)) - q(z)log(p(z))] dz
                 = E_{z \sim q(z)} [log(q(z)) - log(p(z))]
        step1. sample z from agg-post
        step2. logq = E_z log(agg-post(z))
        step3. logp = E_z log(prior(z))
        step4. return (logq-logp).mean()
        
        Args:
            mean (Tensor): [batch_size, dim_z]
            logvar (Tensor): [batch_size, dim_z]
            n_samples (int, optional): the times of replicated estimation
            
        """
        batch_size, dim_z = mean.shape
        if agg_size is None:
            agg_size = batch_size
        #elif not self.training and not batch_size%agg_size==0:
        elif not batch_size%agg_size==0:
            agg_size = batch_size
        assert batch_size%agg_size==0, f"batch_size: {batch_size}, agg_size:{agg_size}"
        
        # the variable names:
        # num_chunks: C in section "Aggregation Size for Ablation", equals to 1 by default
        # agg_size: |b| in section "Aggregation Size for Ablation", equals to |B| by default
        # i.e., |B| = C * |b|
        # n_samples: M in equations
        # i.e., S = M * |B| = M * |b| * C points of DG are calculated and avraged to the final results
        
        num_chunks = int(batch_size/agg_size)
        
        # step1. sample z from agg-post
        z = GaussianReParameterize(mean, logvar, n_samples)
        # z: [batch_size, n_samples, dim_z]
        
        z1 = z.view(num_chunks, agg_size, n_samples, dim_z) # group into different chunks
        z2 = z1.view(num_chunks, agg_size*n_samples, dim_z) # share inside chunks
        z3 = z2.unsqueeze(1).repeat(1,agg_size,1,1) # copy for every posterior inside chunks
        # [num_chunks, agg_size, agg_size*n_samples, dim_z]
        
        # step2. logq = E_z log(agg-post(z)) = E_z log[1/N q(z_{i,j}|x_{k})], where N is agg_size, i in range(N)
        # q(z_{i,j}|x_{k}) \in R^dim_z is the marginal density on each dimension
        q_ij_given_k = GaussianPDF(
            mean.view(num_chunks, agg_size, dim_z), # [num_chunks, agg_size, dim_z]
            logvar.view(num_chunks, agg_size, dim_z), # [num_chunks, agg_size, dim_z]
            z3, # [num_chunks, agg_size, agg_size*n_samples, dim_z]
        ) # the marginal density on each dimension: [num_chunks, agg_size, agg_size*n_samples, dim_z]
        logq = q_ij_given_k.mean(dim=1).log().mean(dim=1) # the first mean is for agg, and the second is for mc: |b| * M
        # logq: [num_chunks, dim_z]
        
        # step3. logp = E_z log(prior(z)) = E_z log[p(z_{i,j})]
        logp_marginal_i = LogGaussianPDF(
            torch.zeros_like(z2[:,0,:]),
            torch.zeros_like(z2[:,0,:]),
            z2, # [num_chunks, agg_size*n_samples, dim_z]
        )
        logp = logp_marginal_i.mean(dim=1) # this mean is for mc
        # logp: [num_chunks, dim_z]
        
        # step4. return (logq-logp).mean()
        okl = (logq-logp).mean(dim=0) # this mean is for mc: C
        return okl.sum(dim=-1) # sum over all dimensions

# implementation of Eq. 9 in paper for vMF distribution
class DGBasedVonMisesFisherKLD(torch.nn.Module):
    """ implementation of KL(agg-post || prior) through Mentor Carlo approximation on Density Gap
    where:
        1. agg-post = mean_i(post_i), post_i \sim vMF(\mu_i, kappa)
        2. prior \sim vMF(0, kappa)
    """
    def __init__(self, vmfKernel):
        super().__init__()
        self.vmfKernel = vmfKernel
        
    def forward(self, mu, n_samples=32, agg_size=None):
        """ KL(agg-post || prior)
        KL(q||p) = \int_z [q(z)log(q(z)) - q(z)log(p(z))] dz
        step1. sample z from agg-post
        step2. logq = log(agg-post(z))
        step3. logp = log(prior(z))
        step4. return (logq-logp).mean()
        
        Args:
            mu (Tensor): [batch_size, dim_z]
            n_samples (int, optional): the times of replicated estimation
            
        """
        batch_size, dim_z = mu.shape
        if agg_size is None:
            agg_size = batch_size
        elif not self.training and not batch_size%agg_size==0:
            agg_size = batch_size
        assert batch_size%agg_size==0, f"batch_size: {batch_size}, agg_size:{agg_size}"
        
        num_chunks = int(batch_size/agg_size)
        
        # step1. sample z from agg-post
        z = self.vmfKernel.ReParameterize(mu, n_samples)
        # [batch_size, n_samples, dim_z]
        
        z1 = z.view(num_chunks, agg_size, n_samples, dim_z) # group into different chunks
        z2 = z1.view(num_chunks, agg_size*n_samples, dim_z) # mix inside chunks
        z3 = z2.unsqueeze(1).repeat(1,agg_size,1,1) # share inside chunks
        z_shared = z3.view(agg_size*num_chunks, agg_size*n_samples, dim_z) # reshape to batch_size
        # [batch_size, agg_size*n_samples, dim_z]
        
        # step2. logq = log(agg-post(z))
        logq = self.vmfKernel.LogPDF(
            mu, # [batch_size, dim_z]
            z_shared, # [batch_size, agg_size*n_samples, dim_z]
        ).view(num_chunks, agg_size, agg_size*n_samples)
        logq = exp_mean_log(logq, dim=1).mean(dim=1) # the first mean is for agg, and the second is for mc
        # logq: [num_chunks]
        
        # step3. logq = log(prior(z))
        logp = self.vmfKernel.log_C_d_zero
        
        # step4. return (logp-logq).mean()
        okl = (logq-logp).mean(dim=0)
        return okl