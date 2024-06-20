import numpy as np
import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                                           unnormalized_derivatives, inverse=False, tails="linear",
                                           tail_bound=1.0, min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                           min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                           min_derivative=DEFAULT_MIN_DERIVATIVE,
                                           enable_identity_init=False):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            enable_identity_init=enable_identity_init
        )

    return outputs, logabsdet


def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                              inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE, enable_identity_init=False):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]
    validate_min_bin_width_height(num_bins, min_bin_width, min_bin_height)

    widths = calculate_widths(unnormalized_widths, min_bin_width, right, left)
    cumwidths = calculate_cumulative_widths(widths, left, right)
    widths = calculate_bin_width(widths, cumwidths)

    derivatives = calculate_derivatives(unnormalized_derivatives, min_derivative)
    heights = calculate_heights(unnormalized_heights, min_bin_height, top, bottom)
    cumheights = calculate_cumulative_heights(heights, bottom, top)
    heights = calculate_bin_height(heights, cumheights)

    if enable_identity_init:
        beta = np.log(2) / (1 - min_derivative)
    else:
        beta = 1

    if inverse:
        bin_idx = torchutils.searchsorted(cumheights, inputs)[..., None]
        bin_cumheights = input_cumheights[..., 0]
        bin_heights = heights.gather(-1, bin_idx)[..., 0]
        bin_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        bin_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
        bin_delta = delta.gather(-1, bin_idx)[..., 0]
        bin_widths = widths.gather(-1, bin_idx)[..., 0]

        a, b, c = calculate_inverse_coefficients(inputs, bin_cumheights, bin_derivatives,
                                                 bin_derivatives_plus_one, bin_delta, bin_heights)
        outputs, logabsdet = calculate_inverse_outputs(inputs, bin_widths, a, b, c)

        return outputs, -logabsdet
    else:
        bin_idx = torchutils.searchsorted(cumwidths, inputs)[..., None]
        bin_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        bin_widths = widths.gather(-1, bin_idx)[..., 0]
        bin_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        bin_heights = heights.gather(-1, bin_idx)[..., 0]
        bin_delta = delta.gather(-1, bin_idx)[..., 0]
        bin_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        bin_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        outputs, logabsdet = calculate_outputs(inputs, bin_cumwidths, bin_widths, bin_cumheights,
                                               bin_heights, bin_delta, bin_derivatives, bin_derivatives_plus_one)

        return outputs, logabsdet


def validate_min_bin_width_height(num_bins, min_bin_width, min_bin_height):
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")


def calculate_widths(unnormalized_widths, min_bin_width, right, left):
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    return widths


def calculate_cumulative_widths(widths, left, right):
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right

    return cumwidths


def calculate_bin_width(widths, cumwidths):
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    return widths


def calculate_derivatives(unnormalized_derivatives, min_derivative):
    derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=beta)

    return derivatives


def calculate_heights(unnormalized_heights, min_bin_height, top, bottom):
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    return heights


def calculate_cumulative_heights(heights, bottom, top):
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top

    return cumheights


def calculate_bin_height(heights, cumheights):
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    return heights


def calculate_inverse_coefficients(inputs, bin_cumheights, bin_derivatives,
                                   bin_derivatives_plus_one, bin_delta, bin_heights):
    a = (inputs - bin_cumheights) * (bin_derivatives + bin_derivatives_plus_one - 2 * bin_delta) + bin_heights * (
        bin_delta - bin_derivatives)
    b = bin_heights * bin_derivatives - (inputs - bin_cumheights) * (
        bin_derivatives + bin_derivatives_plus_one - 2 * bin_delta)
    c = -bin_delta * (inputs - bin_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    return a, b, c


def calculate_inverse_outputs(inputs, bin_widths, a, b, c):
    root = (2 * c) / (-b - torch.sqrt(discriminant))
    outputs = root * bin_widths + bin_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = bin_delta + ((bin_derivatives + bin_derivatives_plus_one - 2 * bin_delta) * theta_one_minus_theta)
    derivative_numerator = bin_delta.pow(2) * (
        bin_derivatives_plus_one * root.pow(2) + 2 * bin_delta * theta_one_minus_theta +
        bin_derivatives * (1 - root).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


def calculate_outputs(inputs, bin_cumwidths, bin_widths, bin_cumheights, bin_heights,
                      bin_delta, bin_derivatives, bin_derivatives_plus_one):
    theta = (inputs - bin_cumwidths) / bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = bin_heights * (
        bin_delta * theta.pow(2) + bin_derivatives * theta_one_minus_theta
    )
    denominator = bin_delta + ((bin_derivatives + bin_derivatives_plus_one - 2 * bin_delta) * theta_one_minus_theta)
    outputs = bin_cumheights + numerator / denominator

    derivative_numerator = bin_delta.pow(2) * (
        bin_derivatives_plus_one * theta.pow(2) + 2 * bin_delta * theta_one_minus_theta +
        bin_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet