import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
The utils for Gaussian-Distribution based VAE
"""
def GaussianPDF(mean, logvar, z):
    r""" Return the PDF value of z in N(mean,exp(logvar))
    mean, logvar : [*, dim_z]
    z : [*, N, dim_z]
    return: [*, N, dim_z]
    """
    if type(mean) is torch.Tensor:
        mean, logvar = mean.unsqueeze(-2), logvar.unsqueeze(-2)
        return 1/(np.sqrt(2*np.pi)*torch.exp(logvar*0.5)) * torch.exp(-((z-mean)**2) / (2*torch.exp(logvar)))
    elif type(mean) is np.ndarray:
        mean, logvar = np.expand_dims(mean, axis=-2), np.expand_dims(logvar, axis=-2)
        return 1/(np.sqrt(2*np.pi)*np.exp(logvar*0.5)) * np.exp(-((z-mean)**2) / (2*np.exp(logvar)))
    return None

def LogGaussianPDF(mean, logvar, z):
    r""" Return the log PDF value of z in N(mean,exp(logvar))
    mean, logvar : [*, dim_z]
    z : [*, N, dim_z]
    return: [*, N, dim_z]
    """
    if type(mean) is torch.Tensor:
        mean, logvar = mean.unsqueeze(-2), logvar.unsqueeze(-2)
        return -0.5*np.log(2*np.pi) -0.5*logvar - ((z-mean)**2+1e-6) / (2*torch.exp(logvar)+1e-6)
    elif type(mean) is np.ndarray:
        mean, logvar = np.expand_dims(mean, axis=-2), np.expand_dims(logvar, axis=-2)
        return -0.5*np.log(2*np.pi) -0.5*logvar - ((z-mean)**2+1e-6) / (2*np.exp(logvar)+1e-6)
    return None

def GaussianReParameterize(mean, logvar, n_samples=1):
    r""" Sampling from N(mean,exp(logvar)) through: z = mean + exp(logvar*0.5)*eps
    mean, logvar : [batch_size, dim_z]
    n_samples : int (default 1)
    return: [batch_size, n_samples, dim_z] if n_samples>1 else [batch_size, dim_z]
    """
    mu = mean
    sigma = torch.exp(logvar*0.5)
    if n_samples>1:
        mu_expd = mu.unsqueeze(1).repeat(1, n_samples, 1)
        sigma_expd = sigma.unsqueeze(1).repeat(1, n_samples, 1)
    else:
        mu_expd, sigma_expd = mu, sigma
    eps = torch.zeros_like(sigma_expd).normal_()
    z = eps*sigma_expd + mu_expd
    return z

def GaussianKLD(mean, logvar, reduction=True):
    if reduction:
        loss_kld = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
    else:
        loss_kld = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
    return loss_kld




"""
The utils for vMF-Distribution based VAE
"""
class vonMisesFisherKernel(object):
    def __init__(self, kappa, dim_z, num_caches=10**7):
        self.kappa = kappa
        self.dim_z = dim_z
        self.num_caches = num_caches
        
        self.calculate_Constants()
        self.calculate_InverseCDF()
    
    def LogPDF(self, mu, z):
        r"""
        mu : [batch_size, dim_z]
        z : [batch_size, N, dim_z]
        return: [batch_size, N]
        """
        return self.log_C_d_kappa + self.kappa*torch.bmm(z, mu.unsqueeze(-1)).squeeze(-1)
    
    def PDF(self, mu, z):
        r""" Return the PDF value of z in vMF(mu, kappa): PDF(z,mu,kappa) = q(z|mu,kappa) = exp(log_C_d(\kappa) + \kappa cos<mu,z>)
        mu : [batch_size, dim_z]
        z : [batch_size, N, dim_z]
        return: [batch_size, N]
        """
        return torch.exp(self.LogPDF(mu, z))
    
    def ReParameterize(self, mu, n_samples=1):
        r""" Sampling fromvMF(mu, kappa) through: z = w*mu + v*nu, where nu \sim eps, nu*nu=1 and mu*nu=0
        mu : [batch_size, dim_z]
        n_samples : int (default 1)
        return: [batch_size, n_samples, dim_z]
        """
        batch_size, dim_z = mu.shape
        
        # sampling w
        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        idxs = (torch.rand([batch_size, n_samples, 1])*self.num_caches).long()
        w, v = self.w[idxs], self.v[idxs]
        
        # sampling nu
        eps = torch.zeros_like(mu).normal_()
        nu = eps - torch.sum(eps*mu, dim=-1).unsqueeze(-1)*mu
        nu = nu / torch.norm(nu, dim=-1).unsqueeze(-1)
        
        z = w*mu + v*nu
        return z
    
    def KLD(self, mu):
        return self.kappa*self.mu_E_q_z + self.log_C_d_kappa - self.log_C_d_zero
        
    def calculate_Constants(self):
        """calculate the constant terms for the computations of PDF and KLD in vMF-VAE:
            KLD(q(z|mu,kappa)||p(z)) = \kappa \mu E_q[z] + log(C_d(\kappa)) - log(C_d(0))
            PDF(z,mu,kappa) = q(z|mu,kappa) = exp(log_C_d(\kappa) + \kappa cos<mu,z>)
        include:
            mu_E_q[z]
            log(C_d(\kappa))
            log(C_d(0))
        where
            E_q[z] = \frac{I_{d/2}(\kappa)}{I_{d/2-1}(\kappa)}
            C_d(x) = \frac{x^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(x)}
        References
            [1] Hyperspherical Variational Auto-Encoders
            [2] Spherical Latent Spaces for Stable Variational Autoencoders
        """
        from scipy import special as sp
        k = self.kappa
        d = self.dim_z
        
        self.mu_E_q_z = sp.iv(d/2, k) / sp.iv(d/2-1, k)
        def logCdx(d, k):
            return (d/2-1)*np.log(k) - np.log(2*np.pi)*d/2 - np.log(sp.iv(d/2-1, k))
        def logCd0(d):
            return -(d/2)*np.log(np.pi) - np.log(2) + sp.loggamma(d/2).real
        self.log_C_d_kappa = logCdx(d, k)
        self.log_C_d_zero = logCd0(d)
        # it can also calculated through:
        # self.log_C_d_zero = -(d/2)*np.log(np.pi) - np.log(2) + sp.loggamma(d/2).real
        # >>> logCdx(10, 1e-10)
        # -3.2387427794590025
        # >>> -(d/2)*np.log(np.pi) - np.log(2) + sp.loggamma(d/2).real
        # -3.2387427794590002
        # however, logCdx(d, k) can return inf when d is large and k is small, e.g. logCdx(768, 1e-10), while logCd0 is more reliable
    
    def calculate_InverseCDF(self):
        """preprocess the following steps to enable quick sampling of w = \cos\varphi \in [-1,1]
        with p(w) \sim e^{\kappa w} (1-w^2)^{(d-3)/2} = exp[\kappa w + log(1-w^2) (d-3)/2]
        
        step 1
        calculate the Cumulative Distribution Function according to the Probability Density Function
        CDF(w) = P(W<=w), where p(w) \sim e^{\kappa w} (1-w^2)^{(d-3)/2} = exp[\kappa w + log(1-w^2) (d-3)/2]
        so, CDF(w) = \int_{-1<x<w} p(w) dx / \int_{-1<x<1} p(w) dx
        
        step 2
        calculate the Inverse CDF through one-dimensional piecewise linear interpolant
        """
        # step 1
        x = np.linspace(-1, 1, self.num_caches+2)[1:-1]
        y = self.kappa*x + np.log(1-x**2)*(self.dim_z-3)/2
        y = np.cumsum(np.exp(y-y.max()))
        y = y/y[-1]
        
        # step 2
        self.InverseCDF = torch.Tensor(np.interp(np.linspace(0, 1, self.num_caches+2)[1:-1], y, x))
        self.w = self.InverseCDF.cuda()
        self.v = np.sqrt(1-self.InverseCDF**2).cuda()
    
    def __repr__(self):
        return f"{self.__class__.__name__}: d={self.dim_z}, k={self.kappa}, log(Cdk)={self.log_C_d_kappa}, log(Cd0)={self.log_C_d_zero}, kld={self.KLD(None)}"