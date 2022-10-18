from utils_old import uniform_initializer, calc_mi, calc_au
import math
import torch
import numpy as np
from tqdm import tqdm

visualize_settings = {
    "x_y_range": (5,5), # how big would a visualization plot be
    "x_y_samples": (500,500), # how many marginal points would be in a visualization plot
}

def GaussianPDF(mu, logvar, z):
    """ Returns the PDF value of z in N(mu,exp(logvar))
    mu, logvar : [batch_size, dim_z]
    z : [batch_size, N, dim_z]
    """
    mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
    return 1/(np.sqrt(2*np.pi)*torch.exp(logvar*0.5)) * torch.exp(-((z-mu)**2) / (2*torch.exp(logvar)))

def StdGaussianPDF(z):
    """ Returns the PDF value of z in N(0,1)
    z : [batch_size, N, dim_z]
    """
    return 1/(np.sqrt(2*np.pi)) * torch.exp(-z**2 / 2)

def get_2D_axis(all_mu, all_logvar, plt, ax, select_axis):
    """ Returns two specific dimensions of the latent vector for visualization
    all_mu, all_logvar: [*, dim_z]
    select_axis can be:
        1. tuple of intergers that indicate the two dimensions, e.g. (15, 31)
        2. string:
            a. "max_var"
            b. "min_var"
            c. "mid_var"
            which denote to choose the two dimensions that have the highest/lowest/middle variance across samples
    """
    dim_z = all_mu.shape[1]
    mu_var = np.var(all_mu, axis=0)
    mu_var_sort = np.argsort(mu_var, axis=0)
    if type(select_axis) is tuple:
        axis_x, axis_y = mu_var_sort[select_axis[0]], mu_var_sort[select_axis[1]]
    if type(select_axis) is list:
        axis_x, axis_y = mu_var_sort[select_axis[0]], mu_var_sort[select_axis[1]]
        plt.text(0, 1, f"   variance:{mu_var[axis_x]:.3f}, rank:{select_axis[0]}", size = 10,
                 family = "fantasy", color = "black", style = "italic", weight = "light",
                      ma = 'left', ha = 'left', va = "top", transform=ax.transAxes)
        plt.text(0, 1, f"   variance:{mu_var[axis_y]:.3f}, rank:{select_axis[1]}", size = 10, rotation=-90,
                 family = "fantasy", color = "black", style = "italic", weight = "light",
                 ma = 'left', ha = 'left', va = "top", transform=ax.transAxes)
    elif select_axis == "max_var": # the axis with the highest variance of mu across samples
        axis_x, axis_y = mu_var_sort[-2], mu_var_sort[-1]
    elif select_axis == "min_var": # the axis with the lowest variance of mu across samples
        axis_x, axis_y = mu_var_sort[0], mu_var_sort[1]
    elif select_axis == "mid_var": # the axis with the middle variance of mu across samples
        axis_x, axis_y = mu_var_sort[int(dim_z/2)-1], mu_var_sort[int(dim_z/2)]
    return axis_x, axis_y

def visualize2D_posterior_distribution(all_mu, all_logvar, plt, ax, mode, select_axis, mask_area=None):
    """ 2D-visulization of given posterior distributions on specific two dimensions
    all_mu, all_logvar: [*, dim_z]
    mode:
        1. "aggregated": visualize the aggregated posterior
        2. "center": visualize the center/mean position of every posterior distribution
    select_axis: please refer to func get_2D_axis
    mask_area: add a semi-translucent mask on specific area, e.g. mask_area=lambda xy:(xy[0]**2+xy[1]**2)>1
    """
    axis_x, axis_y = get_2D_axis(all_mu, all_logvar, plt, ax, select_axis)
    
    if mode == "aggregated":
        visualize2D_aggregated_posterior_distribution(plt, all_mu, all_logvar, axis_x, axis_y, mask_area)
    elif mode == "center":
        visualize2D_center_posterior_distribution(plt, all_mu, all_logvar, axis_x, axis_y, mask_area)
    else:
        assert False, f"unsupported visualization mode: {mode}"
    
    return

def get_marginal_distribution(mu, logvar, axis_x, axis_y):
    """ Returns the marginal probability distributions of N(z;mu, exp(2*logvar)) on specific dimensions for every datapoint
    mu, logvar: [*, dim_z]
    axis_x, axis_y: int \in range(dim_z)
    """
    x_y_range, x_y_samples = visualize_settings["x_y_range"], visualize_settings["x_y_samples"]
    # (*,1,x_range)
    x = np.expand_dims(np.linspace(-x_y_range[0], x_y_range[0], x_y_samples[0]), axis=0)
    mu_x, sigma_x = np.expand_dims(mu[:,axis_x], axis=1), np.expand_dims(np.exp(logvar[:,axis_x]*0.5), axis=1)
    p_x = np.expand_dims((1/(np.sqrt(np.pi*2)*sigma_x)) * np.exp(-((x-mu_x)**2)/(2*(sigma_x**2))), axis=1)
    
    # (*,y_range,1)
    y = np.expand_dims(np.linspace(-x_y_range[1], x_y_range[1], x_y_samples[1]), axis=0)
    mu_y, sigma_y = np.expand_dims(mu[:,axis_y], axis=1), np.expand_dims(np.exp(logvar[:,axis_y]*0.5), axis=1)
    p_y = np.expand_dims((1/(np.sqrt(np.pi*2)*sigma_y)) * np.exp(-((x-mu_y)**2)/(2*(sigma_y**2))), axis=2)
    
    X, Y = np.meshgrid(np.squeeze(x, 0), np.squeeze(y, 0))
    return X, Y, p_x, p_y, None

def visualize2D_center_posterior_distribution(plt, mu, logvar, axis_x, axis_y, mask_area=None):
    """ 2D-visulization of center of every posterior distribution on specific two dimensions
    mu, logvar: [*, dim_z]
    axis_x, axis_y: int \in range(dim_z)
    """
    x_y_range, x_y_samples = visualize_settings["x_y_range"], visualize_settings["x_y_samples"]
    sigma = 0.01 # visualize z \sim N(mu, 0.01^2) instead
    radius_outside = 0.1
    radius_inside = 0.06
    logvar = np.ones_like(logvar) * 2*np.log(sigma)
    X, Y, p_x, p_y, Z = get_marginal_distribution(mu, logvar, axis_x, axis_y)
    
    norm_min = 1/(np.sqrt(np.pi*2)*sigma) * np.exp(-(radius_outside**2)/(2*(sigma**2)))
    norm_max = 1/(np.sqrt(np.pi*2)*sigma) * np.exp(-(radius_inside**2)/(2*(sigma**2)))
    Z = p_x*p_y if Z is None else Z
    Z[Z>=norm_max] = -2
    Z[Z>=norm_min] = -1
    Z[Z>=0] = -3
    Z = (Z+3) / 2
    # 0 --- norm_min --- norm_max --- inf
    #   0        1        0.5
    Z = Z + (np.arange(Z.shape[0])[:,None,None]*(Z>0)*2)
    Z = np.max(Z, axis=0) % 2
    
    import matplotlib
    camp, norm = plt.get_cmap('Blues'), matplotlib.colors.Normalize(0,1)
    plt.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap=camp, norm=norm)
    #plt.colorbar()
    
    if mask_area is not None:
        mask = np.zeros(x_y_samples)*np.nan
        mask[mask_area((X,Y))] = Z.max() - 1e-5
        plt.imshow(mask, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=0.5, origin='lower', cmap=camp, norm=norm)
        
    return
    
def visualize2D_aggregated_posterior_distribution(plt, mu, logvar, axis_x, axis_y, mask_area=None):
    """ 2D-visulization of aggregated posterior distribution on specific two dimensions
    mu, logvar: [*, dim_z]
    axis_x, axis_y: int \in range(dim_z)
    """
    x_y_range, x_y_samples = visualize_settings["x_y_range"], visualize_settings["x_y_samples"]
    X, Y, p_x, p_y, Z = get_marginal_distribution(mu, logvar, axis_x, axis_y)
    Z = p_x*p_y if Z is None else Z
    Z = np.mean(Z, axis=0)
    
    import matplotlib
    camp, norm = plt.get_cmap('hot'), matplotlib.colors.Normalize(0.0,0.25)
    plt.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap=camp, norm=norm)
    plt.colorbar()
    
    if mask_area is not None:
        mask = np.zeros(x_y_samples)*np.nan
        mask[mask_area((X,Y))] = 0.25 - 1e-5
        plt.imshow(mask, extent=[X.min(), X.max(), Y.min(), Y.max()], alpha=0.5, origin='lower', cmap=camp, norm=norm)
        
    return

class pltManager:
    """ An encapsulation of matplotlib.pyplot for automatic subplot allocation
    """
    def __init__(self, plt, columns, lines, figsize=None):
        self.plt = plt
        self.plt.figure(figsize=figsize if figsize else (4*columns, 3*lines))
        self.columns = columns
        self.lines = lines
        self.index = 0
    
    def subplot(self):
        """ return the next subplot in form of ax
        """
        self.index += 1
        ax = self.plt.subplot(self.lines, self.columns, self.index)
        self.plt.subplots_adjust(hspace=0.35)
        return ax