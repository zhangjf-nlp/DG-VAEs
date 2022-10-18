import math
import torch
import torch.nn as nn

from .utils import *

class EncoderBase(nn.Module):
    """docstring for EncoderBase"""
    def __init__(self):
        super(EncoderBase, self).__init__()
    
    def input_to_posterior(self, x):
        """
        Args:
            x [batch_size, input_len]: the token ids of input sentences
        Returns: Tensor1, Tensor2
            Tensor1 [batch_size, dim_z]: the mean of posterior
            Tensor2 [batch_size, dim_z]: the logvar of posterior (meaningless for vMF-based VAE)
        """
        raise NotImplementedError
    
    def input_to_zs(self, x, nsamples):
        """
        Args:
            x [batch_size, input_len]: the token ids of input sentences
        Returns: Tensor1
            Tensor1 [batch_size, nsamples, dim_z]: the zs sampled from posterior
        """
        raise NotImplementedError
    
    def posterior_to_kl(self, mean, logvar):
        """
        Args:
            mean [batch_size, dim_z]: the mean of posterior
            logvar [batch_size, dim_z]: the logvar of posterior (meaningless for vMF-based VAE)
        Returns: Tensor1
            Tensor1 []: the loss_kld for training
        """
        raise NotImplementedError
        
    def posterior_to_zs(self, mean, logvar, nsamples):
        """
        Args:
            mean [batch_size, dim_z]: the mean of posterior
            logvar [batch_size, dim_z]: the logvar of posterior (meaningless for vMF-based VAE)
        Returns: Tensor1
            Tensor1 [batch_size, nsamples, dim_z]: the sampled zs from posterior
        """
        raise NotImplementedError
        
    def prior_to_zs(self, zeros, nsamples):
        """
        Args:
            zeros [batch_size, dim_z]: indicates the prior, e.g. N(0,exp(0))
        Returns: Tensor1
            Tensor1 [batch_size, nsamples, dim_z]: the sampled zs from prior
        """
        raise NotImplementedError
    
    def forward(self, x):
        """
        Args:
            x [batch_size, input_len]: the token ids of input sentences
        Returns: Tensor1, Tensor2, Tensor3, Tensor4
            Tensor1 [batch_size, dim_z]: the mean of posterior
            Tensor2 [batch_size, dim_z]: the logvar of posterior (meaningless for vMF-based VAE)
            Tensor3 [batch_size, dim_z]: the single z (for each input) sampled from posterior
            Tensor4 []: the loss_kld for training
        """
        raise NotImplementedError

class LSTMEncoder(EncoderBase):
    """
    implemented:
        x -> mean, logvar
    not implemented:
        mean, logvar -> z
        mean, logvar -> kl
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = nn.LSTM(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=1, batch_first=True, dropout=0)
        self.hidden_state_to_posterior = nn.Linear(args.hidden_size, 2*args.dim_z, bias=False)
        if issubclass(LSTMEncoder, self.__class__): # False in subclass
            self.reset_parameters(model_init, emb_init)
    
    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embedding.weight)
    
    def input_to_posterior(self, x):
        embeddings = self.embedding(x)
        hidden_states, (last_hidden_state, last_cell_state) = self.lstm(embeddings)
        mean, logvar = self.hidden_state_to_posterior(last_hidden_state).chunk(2, -1)
        #print(f"mean.shape: {mean.shape}") # torch.Size([1, batch_size, dim_z])
        return mean.squeeze(0), logvar.squeeze(0)
    
    def input_to_zs(self, x, nsamples):
        mean, logvar = self.input_to_posterior(x)
        zs = self.posterior_to_zs(mean, logvar, nsamples)
        return zs
        
    def posterior_to_zs(self, mean, logvar, nsamples):
        raise NotImplementedError
        
    def prior_to_zs(self, zeros, nsamples):
        raise NotImplementedError
        
    def posterior_to_kl(self, mean, logvar):
        raise NotImplementedError
    
    def forward(self, x):
        mean, logvar = self.input_to_posterior(x)
        z = self.posterior_to_zs(mean, logvar, nsamples=1)
        loss_kld = self.posterior_to_kl(mean, logvar)
        return mean, logvar, z, loss_kld

class GaussianLSTMEncoder(LSTMEncoder):
    """
    implemented:
        x -> mean, logvar
        mean, logvar -> z (new)
        mean, logvar -> kl (new)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.useGaussian = True
        if issubclass(GaussianLSTMEncoder, self.__class__): # False in subclass
            self.reset_parameters(model_init, emb_init)
    
    def posterior_to_zs(self, mean, logvar, nsamples):
        batch_size, dim_z = mean.shape
        expanded_mean = mean.unsqueeze(1).expand(batch_size, nsamples, dim_z)
        expanded_logvar = logvar.unsqueeze(1).expand(batch_size, nsamples, dim_z)
        epsilons = torch.zeros_like(expanded_mean).normal_()
        zs = expanded_mean + torch.exp(expanded_logvar*0.5)*epsilons
        return zs
    
    def prior_to_zs(self, zeros, nsamples):
        return self.posterior_to_zs(zeros, zeros, nsamples)
        
    def posterior_to_kl(self, mean, logvar):
        loss_kld = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        return loss_kld
    
    def prior_density_log_factor(self, mean, logvar, zs, reverse=False):
        """ return log(p(z)/q(z)), this is used for more accurate MC estimation on prior/post log likelihood
            where p(z) = prior(z) (not reverse)
                          post(z) (reverse)
            and   q(z) = (prior(z)+post(z))/2
        mean, logvar : [batch_size, dim_z]
        zs : [batch_size, N, dim_z]
        return: [batch_size, N]
        """
        log_dPost_z = LogGaussianPDF(
            mean, logvar, # [batch_size, dim_z]
            zs, # [batch_size, 2*n_samples, dim_z]
        ).sum(dim=-1)
        log_dPrior_z = LogGaussianPDF(
            torch.zeros_like(mean), torch.zeros_like(logvar), # [batch_size, dim_z]
            zs, # [batch_size, 2*n_samples, dim_z]
        ).sum(dim=-1)
        base, _ = torch.max(torch.cat((log_dPrior_z.unsqueeze(-1),log_dPost_z.unsqueeze(-1)),dim=-1),dim=-1)
        log_dQ_z = torch.log(((log_dPost_z-base).exp()+(log_dPrior_z-base).exp())/2) + base
        log_dP_z = log_dPost_z if reverse else log_dPrior_z
        return log_dP_z - log_dQ_z

""" VAE variants for solving kl-vanishing, referring to https://github.com/valdersoul/bn-vae """
class BNGaussianLSTMEncoder(GaussianLSTMEncoder):
    """
    implemented:
        x -> mean, logvar (modified)
        mean, logvar -> z
        mean, logvar -> kl
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.bn = nn.BatchNorm1d(args.dim_z)
        self.bn.weight.requires_grad = False
        self.reset_parameters(model_init, emb_init)
        self.bn.weight.fill_(args.gamma)
        
    def input_to_posterior(self, x):
        mean, logvar = super().input_to_posterior(x)
        mean = self.bn(mean)
        return mean, logvar

class DeltaGaussianLSTMEncoder(GaussianLSTMEncoder):
    """
    implemented:
        x -> mean, logvar (modified)
        mean, logvar -> z
        mean, logvar -> kl
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        # EQ 4 in the delta-vae paper
        from sympy import Symbol, solve, ln, sqrt
        x = Symbol('x')
        l_var, u_var = solve([ln(x)-x + 2*self.args.delta + 1],[x])
        l_std, u_std = sqrt(l_var[0]), sqrt(u_var[0])
        self.l = torch.tensor(float(l_std), device=self.args.device)
        self.u = torch.tensor(float(u_std), device=self.args.device)
        self.reset_parameters(model_init, emb_init)
    
    def input_to_posterior(self, x):
        mean, logvar = super().input_to_posterior(x)
        std = self.l + (self.u - self.l) * (1 / torch.clamp((1 + torch.exp(-logvar)), 1, 50))
        logvar = torch.log(std**2)
        mean = torch.sqrt(2 * self.args.delta + 1 + logvar - torch.exp(logvar)
                          + torch.max(torch.tensor(0.0, device=self.args.device), mean) + 1e-6)
        return mean, logvar

class FineFBGaussianLSTMEncoder(GaussianLSTMEncoder):
    """
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.reset_parameters(model_init, emb_init)
    
    def posterior_to_kl(self, mean, logvar):
        loss_kld = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_mask = (loss_kld > (self.args.target_kl/self.args.dim_z)).float()
        loss_kld = (kl_mask * loss_kld).sum(dim=1).mean(dim=0)
        return loss_kld

class CoarseFBGaussianLSTMEncoder(GaussianLSTMEncoder):
    """
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.reset_parameters(model_init, emb_init)
    
    def posterior_to_kl(self, mean, logvar):
        loss_kld = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        kl_mask = (loss_kld > self.args.target_kl).float()
        loss_kld = (kl_mask * loss_kld).mean(dim=0)
        return loss_kld

class VMFLSTMEncoder(LSTMEncoder):
    """
    implemented:
        x -> mean, logvar (modified)
        mean, logvar -> z (new)
        mean, logvar -> kl (new)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.useGaussian = False
        from .utils import vonMisesFisherKernel
        self.vmfKernel = vonMisesFisherKernel(args.kappa, args.dim_z)
        if issubclass(VMFLSTMEncoder, self.__class__): # False in subclass
            self.reset_parameters(model_init, emb_init)
    
    def input_to_posterior(self, x):
        mean, logvar = super().input_to_posterior(x)
        mean = torch.nn.functional.normalize(mean, dim=-1)
        return mean, torch.ones_like(mean) * 1000
    
    def posterior_to_zs(self, mean, logvar, nsamples):
        zs = self.vmfKernel.ReParameterize(mean, nsamples)
        return zs
    
    def prior_to_zs(self, zeros, nsamples):
        batch_size, dim_z = zeros.shape
        epsilons = zeros.repeat(nsamples, 1).normal_() # [batch_size*nsamples, dim_z]
        mean = torch.nn.functional.normalize(epsilons, dim=-1)
        return self.posterior_to_zs(mean, None, 1).view(batch_size, nsamples, dim_z)
    
    def posterior_to_kl(self, mean, logvar):
        loss_kld = self.vmfKernel.KLD(mean)
        return loss_kld
    
    def prior_density_log_factor(self, mean, logvar, zs, reverse=False):
        """ return log(p(z)/q(z)), this is used for more accurate MC estimation on prior/post log likelihood
            where p(z) = prior(z) (not reverse)
                          post(z) (reverse)
            and   q(z) = (prior(z)+post(z))/2
        mean, logvar : [batch_size, dim_z]
        zs : [batch_size, N, dim_z]
        return: [batch_size, N]
        """
        # return p(z) / q(z)
        # where p(z) is prior(z) (not reverse)
        #               post(z) (reverse)
        # q(z) = (prior(z)+post(z))/2
        log_dPost_z = self.vmfKernel.LogPDF(
            mean, # [batch_size, dim_z]
            zs, # [batch_size, 2*n_samples, dim_z]
        ) # [batch_size, 2*n_samples]
        log_dPrior_z = torch.ones_like(log_dPost_z) * self.vmfKernel.log_C_d_zero  # [batch_size, 2*n_samples]
        base, _ = torch.max(torch.stack([log_dPost_z,log_dPrior_z],dim=-1),dim=-1) # prevent exp overflow
        log_dQ_z = torch.log(((log_dPost_z-base).exp()+(log_dPrior_z-base).exp())/2) + base
        log_dP_z = log_dPost_z if reverse else log_dPrior_z
        return log_dP_z - log_dQ_z

""" AAE and WAE """
class AAEGaussianLSTMEncoder(GaussianLSTMEncoder):
    """ Adversarial Auto-Encoder, referring to https://github.com/Naresh1318/Adversarial_Autoencoder
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.discriminator = nn.Sequential(
            nn.Linear(args.dim_z, args.dim_z),
            nn.Sigmoid(),
            nn.Linear(args.dim_z, args.dim_z),
            nn.Sigmoid(),
            nn.Linear(args.dim_z, 2),
        )
        self.reset_parameters(model_init, emb_init)
    
    def discriminator_detach_forward(self, zs):
        w1, b1 = self.discriminator[0].weight.detach(), self.discriminator[0].bias.detach()
        sigmoid1 = self.discriminator[1]
        w2, b2 = self.discriminator[2].weight.detach(), self.discriminator[2].bias.detach()
        sigmoid2 = self.discriminator[3]
        w3, b3 = self.discriminator[4].weight.detach(), self.discriminator[4].bias.detach()
        #return torch.mm(sigmoid(torch.mm(zs, w1.T) + b1[None,:]), w2.T) + b2[None,:]
        return torch.mm(sigmoid2(torch.mm(sigmoid1(torch.mm(zs, w1.T) + b1[None,:]), w2.T) + b2[None,:]), w3.T) + b3[None,:]
    
    def posterior_to_kl(self, mean, logvar, nsamples=32):
        batch_size, dim_z = mean.shape
        prior_zs = self.prior_to_zs(torch.zeros_like(mean), nsamples).view(batch_size*nsamples, dim_z)
        post_zs = self.posterior_to_zs(mean, logvar, nsamples).view(batch_size*nsamples, dim_z)
        loss_dis = F.cross_entropy(self.discriminator(torch.cat([prior_zs, post_zs], dim=0).detach()),
                                   torch.cat([torch.ones_like(prior_zs[:,0]), torch.zeros_like(post_zs[:,0])], dim=0).long())
        loss_gen = F.cross_entropy(self.discriminator_detach_forward(post_zs),
                                   torch.ones_like(post_zs[:,0]).long())
        loss_kld = loss_gen + loss_dis
        return loss_kld

class WAEGaussianLSTMEncoder(GaussianLSTMEncoder):
    """ Wasserstein Auto-Encoder, referring to https://github.com/1Konny/WAE-pytorch
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        self.reset_parameters(model_init, emb_init)
    
    def loss_kld(self, mean, logvar, nsamples=32):
        def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
            r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
            Args:
                z1 (Tensor): batch of samples from a multivariate gaussian distribution \
                    with scalar variance of z_var.
                z2 (Tensor): batch of samples from another multivariate gaussian distribution \
                    with scalar variance of z_var.
                exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
            """
            assert z1.size() == z2.size()
            assert z1.ndimension() == 2
            z_dim = z1.size(1)
            C = 2*z_dim*z_var
            z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
            z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)
            kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
            kernel_sum = kernel_matrix.sum()
            # numerically identical to the formulation. but..
            if exclude_diag:
                kernel_sum -= kernel_matrix.diag().sum()
            return kernel_sum
        def mmd(z_tilde, z, z_var=1):
            r"""Calculate maximum mean discrepancy described in the WAE paper.
            Args:
                z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
                    2D Tensor(batch_size x dimension).
                z (Tensor): samples from prior distributions. same shape with z_tilde.
                z_var (Number): scalar variance of isotropic gaussian prior P(Z).
            """
            assert z_tilde.size() == z.size()
            assert z.ndimension() == 2
            n = z.size(0)
            out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
                  im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
                  -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)
            return out
        batch_size, dim_z = mean.shape
        prior_zs = self.prior_to_zs(torch.zeros_like(mean), nsamples).view(batch_size*nsamples, dim_z)
        post_zs = self.posterior_to_zs(mean, logvar, nsamples).view(batch_size*nsamples, dim_z)
        loss_kld = mmd(prior_zs, post_zs)
        return loss_kld
    
""" Proposed Methods """
class DGGaussianLSTMEncoder(GaussianLSTMEncoder):
    """
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        from .DGBasedKLD import DGBasedGaussianKLD
        self.kld = DGBasedGaussianKLD()
        self.reset_parameters(model_init, emb_init)
    
    def posterior_to_kl(self, mean, logvar):
        loss_kld = self.kld(mean, logvar, agg_size=self.args.agg_size)
        return loss_kld

class DGVMFLSTMEncoder(VMFLSTMEncoder):
    """
    implemented:
        x -> mean, logvar
        mean, logvar -> z
        mean, logvar -> kl (modified)
    """
    def __init__(self, args, model_init, emb_init):
        super().__init__(args, model_init, emb_init)
        from .DGBasedKLD import DGBasedVonMisesFisherKLD
        self.kld = DGBasedVonMisesFisherKLD(self.vmfKernel)
        self.reset_parameters(model_init, emb_init)
    
    def posterior_to_kl(self, mean, logvar):
        loss_kld = self.kld(mean, agg_size=self.args.agg_size)
        return loss_kld

available_encoder_classes = {encoder_class.__name__:encoder_class for encoder_class in [
    GaussianLSTMEncoder, # VAE
    BNGaussianLSTMEncoder, # BN-VAE
    DeltaGaussianLSTMEncoder, # Delta-VAE
    FineFBGaussianLSTMEncoder, # FB-VAE (Kingma et al.)
    CoarseFBGaussianLSTMEncoder, # another kind of FB-VAE
    VMFLSTMEncoder, # vMF-VAE
    AAEGaussianLSTMEncoder, # AAE
    WAEGaussianLSTMEncoder, # WAE
    DGGaussianLSTMEncoder, # DG-VAE
    DGVMFLSTMEncoder, # DG-vMF-VAE
]}