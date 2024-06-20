import numpy as np
from houghvst.estimation import gat

def compare_variance_stabilization(img, img_noisy, sigma_gt, alpha_gt, sigma_est, alpha_est):
    assess_variance_stabilization(img, img_noisy, sigma_gt, alpha_gt, heading='Ground truth')
    assess_variance_stabilization(img, img_noisy, sigma_est, alpha_est)

def assess_variance_stabilization(img, img_noisy, params, heading='Estimated'):
    alpha = params.get('alpha', 1.0)
    sigma = params.get('sigma', 0)
    
    img = alpha * img
    
    img_gat = gat.compute_gat(img, sigma, alpha=alpha)
    img_noisy_gat = gat.compute_gat(img_noisy, sigma, alpha=alpha)
    diff = img_gat - img_noisy_gat

    variance = np.var(diff, ddof=1)
    print('--->', heading, 'variance', variance)
    return variance

def compute_temporal_mean_var(movie):
    means, variances = np.mean(movie, axis=0), np.var(movie, axis=0, ddof=1)
    return means, variances