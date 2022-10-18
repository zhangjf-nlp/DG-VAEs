import math
import torch
import torch.nn as nn

from .utils import log_sum_exp, exp_mean_log

class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        mean = torch.zeros(args.dim_z, device=args.device)
        variance = torch.ones(args.dim_z, device=args.device)
        self.prior = torch.distributions.normal.Normal(mean, variance)
    
    def forward(self, x, kl_weight):
        """
        Args:
            x [batch_size, input_len]: the token ids of input sentences
            kl_weight float: the weight of loss_kld in loss_total
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1 []: the loss_total for training
            Tensor2 []: the loss_rec for training
            Tensor3 []: the loss_kld for training
        """
        mean, logvar, z, loss_kld = self.encoder(x)
        loss_kld = loss_kld * self.args.kl_beta
        loss_rec = self.decoder(x, z).mean(dim=0)
        loss_total = loss_rec + loss_kld * kl_weight
        return loss_total, loss_rec, loss_kld
        
    def encode(self, x, nsamples=1):
        """
        Args:
            x [batch_size, input_len]: the token ids of input sentences
        Returns: Tensor1, Tensor2
            Tensor1 [batch_size, nsamples, dim_z]: the sampled zs from posterior
            Tensor2 []: the loss_kld for training
        """
        mean, logvar = self.encoder.input_to_posterior(x)
        zs = self.encoder.posterior_to_zs(mean, logvar, nsamples)
        loss_kld = self.encoder.posterior_to_kl(mean, logvar)
        return zs, loss_kld
    
    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z [batch_size, nsamples, dim_z]: the sampled zs from posterior
            strategy str: "beam" or "greedy" or "sample"
            K int: the beam width for beam-search
        Returns: List1
            List1 list[list[int]]: a list of decoded word sequence
        """
        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")
    
    def evaluate(self, x):
        """evaluate the model through the following metrics
        loss = loss_rec + loss_kld (self.forward)
        elbo = loss_rec + kl
        kl   = \mean_{x_i \in x} {KL(post(z|x_i)||prior(z))}
        okl  = KL(\mean_{x_i \in x}{post(z|x_i)} || prior(z)) (sampling-based) (only DG)
        logp_prior = \mean_{x_i \in x} {E_{z \sim prior(z)}{log p(x,z)}} (sampling-based)
        logp_post  = \mean_{x_i \in x} {E_{z \sim post(z)}{log p(x,z)}} (sampling-based)
        """
        batch_size, seq_len = x.shape
        
        mean, logvar, z, loss_kld = self.encoder(x) # logvar can be None for vmf-encoder
        loss_rec = self.decoder(x, z) # [batch_size, nsamples]
        
        if "Gaussian" in self.encoder.__class__.__name__:
            kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        else:
            assert "VMF" in self.encoder.__class__.__name__
            kl = torch.tensor(self.encoder.vmfKernel.KLD(mean))
        
        if "DG" in self.encoder.__class__.__name__:
            okl = loss_kld
        else:
            okl = torch.tensor(-100)
        
        elbo = - kl - loss_rec.mean(dim=0)
        
        post_zs = self.encoder.posterior_to_zs(mean, logvar, nsamples=8) #[batch_size, nsamples, dim_z]
        #logp_post = self.decoder.log_probability(x, post_zs).exp().mean(dim=1).log().mean(dim=0)
        #logp_post = exp_mean_log(self.decoder.log_probability(x, post_zs)).mean(dim=0)
        
        prior_zs = self.encoder.prior_to_zs(mean, nsamples=8) #[batch_size, nsamples, dim_z]
        #logp_prior = self.decoder.log_probability(x, prior_zs).exp().mean(dim=1).log().mean(dim=0)
        #logp_prior = exp_mean_log(self.decoder.log_probability(x, prior_zs)).mean(dim=0)
        
        # for better estimation of E_{z \sim Prior(z)} [p(x|z)], or E_{z \sim Post(z)} [p(x|z)]
        # we consider:
        # \int p(x|z) d(P(z)) = \int p(x|z)p(z) dz = \int p(x,z) dz = E_{z \sim P(z)} [p(x|z)]
        #                     = \int p(x|z)p(z)/q(z) d(Q(z))
        # where P(z) can be either Prior(z) or Post(z)
        # and Q(z) = (Prior(z)+Post(z)) / 2
        
        # log[p(x|z)post(z)] -- biased
        logp_posts = self.decoder.log_probability(x, post_zs) # [batch_size, nsamples]
        # log[p(x|z)prior(z)] -- biased
        logp_priors = self.decoder.log_probability(x, prior_zs) # [batch_size, nsamples]
        
        # log[p(x|z)(prior(z)+post(z))]
        logp_dQ = torch.cat((logp_priors,logp_posts), dim=1)
        Q_zs = torch.cat((prior_zs,post_zs), dim=1)
        
        log_factor_q_to_prior = self.encoder.prior_density_log_factor(mean, logvar, Q_zs) # [batch_size, nsamples*2]
        logp_prior = exp_mean_log(logp_dQ+log_factor_q_to_prior).mean(dim=0)
        
        log_factor_q_to_post = self.encoder.prior_density_log_factor(mean, logvar, Q_zs, reverse=True) # [batch_size, nsamples*2]
        logp_post = exp_mean_log(logp_dQ+log_factor_q_to_post).mean(dim=0)
        loss_eval = loss_kld*self.args.kl_beta + loss_rec.mean(dim=0)
        
        assert logp_post.item()==logp_post.item()
        
        return [metric_tensor.item() for metric_tensor in 
                [loss_eval, elbo, kl, okl, logp_prior, logp_post]]