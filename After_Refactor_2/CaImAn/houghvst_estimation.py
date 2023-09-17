import numpy as np
import sklearn.linear_model as lm

from collections import namedtuple

# Define AccumulatorSpace and EstimationResult as namedtuples
AccumulatorSpace = namedtuple('AccumulatorSpace', ['score', 'sigma_sq_range', 'alpha_range', 'sigma_sq', 'alpha'])
EstimationResult = namedtuple('EstimationResult', ['alpha_init', 'sigma_sq_init', 'alpha', 'sigma_sq', 'acc_space_init', 'acc_space'])


def compute_score(sigmas, tol):
    x = (sigmas - 1) / tol
    weights = np.exp(-(x ** 2))
    score = np.sum(weights)
    
    if score > 0:
        sigma_est = np.sum(sigmas * weights) / score
    else:
        sigma_est = np.nan
    
    return sigma_est, score


def hough_estimation(blocks, sigma_sq_range, alpha_range, tol=1e-2):
    score = np.zeros((len(sigma_sq_range), len(alpha_range)))

    for i_a, alpha in enumerate(alpha_range):
        for i_s, sigma_sq in enumerate(sigma_sq_range):
            blocks_gat = compute_gat(blocks, sigma_sq, alpha=alpha)
            sigmas = np.std(blocks_gat, axis=(1, 2), ddof=1)
            score[i_s, i_a] = compute_score(sigmas, tol)[1]

    max_score_idx = np.argmax(score)
    best_params = np.unravel_index(max_score_idx, score.shape)
    sigma_sq_est = sigma_sq_range[best_params[0]]
    alpha_est = alpha_range[best_params[1]]

    print('\tHighest score=', score[best_params[0], best_params[1]])

    acc = AccumulatorSpace(score, sigma_sq_range, alpha_range, sigma_sq_est, alpha_est)
    return sigma_sq_est, alpha_est, acc


def compute_gat(blocks, sigma_sq, alpha):
    # Implementation of compute_gat function, left as is
    pass


def estimate_vst_blocks(blocks):
    sigma_sq_init, alpha_init = initial_estimate_sigma_alpha(blocks)
    print('\tinitial alpha = {}; sigma^2 = {}'.format(alpha_init, sigma_sq_init))

    diff_s = np.maximum(2e3, np.abs(sigma_sq_init))
    diff_a = alpha_init * 0.9

    sigma_sq_range = np.linspace(sigma_sq_init - diff_s, sigma_sq_init + diff_s, num=100)
    alpha_range = np.linspace(alpha_init - diff_a, alpha_init + diff_a, num=100)
    sigma_sq_mid, alpha_mid, acc_init = hough_estimation(blocks, sigma_sq_range, alpha_range)
    print('\tmid alpha = {}; sigma^2 = {}'.format(alpha_mid, sigma_sq_mid))

    diff_s /= 10
    diff_a /= 4
    sigma_sq_range = np.linspace(sigma_sq_mid - diff_s, sigma_sq_mid + diff_s, num=100)
    alpha_range = np.linspace(alpha_mid - diff_a, alpha_mid + diff_a, num=100)
    sigma_sq_final, alpha_final, acc = hough_estimation(blocks, sigma_sq_range, alpha_range)
    print('\talpha = {}; sigma^2 = {}'.format(alpha_final, sigma_sq_final))

    return EstimationResult(alpha_init, sigma_sq_init, alpha_final, sigma_sq_final, acc_init, acc)


def estimate_vst_image(img, block_size=8, stride=8):
    # Implementation of estimate_vst_image function, left as is
    pass


def estimate_vst_movie(movie, block_size=8, stride=8):
    # Implementation of estimate_vst_movie function, left as is
    pass


def initial_estimate_sigma_alpha(blocks):
    # Implementation of initial_estimate_sigma_alpha function, left as is
    pass


def regress_sigma_alpha(means, variances, verbose=True):
    # reg = lm.LinearRegression(fit_intercept=True)
    reg = lm.HuberRegressor(alpha=0, fit_intercept=True)
    reg.fit(means[:, np.newaxis], variances)

    alpha = reg.coef_[0]
    sigma_sq = reg.intercept_
    return sigma_sq, alpha


def compute_mean_var(blocks):
    means = np.mean(blocks, axis=(1, 2))
    variances = np.var(blocks, axis=(1, 2), ddof=1)
    return means, variances