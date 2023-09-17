import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3


def unconstrained_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    tail_bound=1.0,
    tails="linear",
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    num_bins = unnormalized_widths.shape[-1]

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
        assert unnormalized_heights.shape[-1] == num_bins - 1
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )

    return outputs, logabsdet


def quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):
    check_inputs_inside_domain(inputs, left, right)

    inputs = normalize_inputs(inputs, inverse, left, right, bottom, top)

    num_bins = unnormalized_widths.shape[-1]

    check_min_bin_sizes(min_bin_width, min_bin_height, num_bins)

    widths = calculate_widths(unnormalized_widths, min_bin_width, num_bins)

    unnorm_heights_exp = calculate_normalized_heights(unnormalized_heights)

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        unnorm_heights_exp = set_boundary_heights(unnorm_heights_exp, widths)

    unnormalized_area = calculate_unnormalized_area(unnorm_heights_exp, widths)
    heights = calculate_heights(unnorm_heights_exp, unnormalized_area, min_bin_height)

    bin_left_cdf = calculate_bin_left_cdf(heights, widths)

    bin_locations = calculate_bin_locations(widths)

    if inverse:
        bin_idx = search_inverse_bin(bin_left_cdf, inputs)
    else:
        bin_idx = search_forward_bin(bin_locations, inputs)

    input_bin_locations, input_bin_widths, input_left_cdf = gather_input_values(bin_locations, widths, bin_left_cdf, bin_idx)

    input_left_heights, input_right_heights = gather_height_values(heights, bin_idx)

    a, b, c = calculate_quadratic_coefficients(input_left_heights, input_right_heights, input_bin_widths, input_left_cdf)

    outputs, logabsdet = calculate_outputs(inputs, inverse, a, b, c, input_bin_widths, input_left_heights, input_right_heights)

    outputs = denormalize_outputs(outputs, inverse, left, right, bottom, top)

    return outputs, logabsdet


def check_inputs_inside_domain(inputs, left, right):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()


def normalize_inputs(inputs, inverse, left, right, bottom, top):
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)
    return inputs


def check_min_bin_sizes(min_bin_width, min_bin_height, num_bins):
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")


def calculate_widths(unnormalized_widths, min_bin_width, num_bins):
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    return widths


def calculate_normalized_heights(unnormalized_heights):
    return F.softplus(unnormalized_heights) + 1e-3


def set_boundary_heights(unnorm_heights_exp, widths):
    first_widths = 0.5 * widths[..., 0]
    last_widths = 0.5 * widths[..., -1]
    numerator = (
        0.5 * first_widths * unnorm_heights_exp[..., 0]
        + 0.5 * last_widths * unnorm_heights_exp[..., -1]
        + torch.sum(
            ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
            * widths[..., 1:-1],
            dim=-1,
        )
    )
    constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
    constant = constant[..., None]
    unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)
    return unnorm_heights_exp


def calculate_unnormalized_area(unnorm_heights_exp, widths):
    return torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths, dim=-1)[..., None]


def calculate_heights(unnorm_heights_exp, unnormalized_area, min_bin_height):
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights
    return heights


def calculate_bin_left_cdf(heights, widths):
    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode='constant', value=0.0)
    return bin_left_cdf


def calculate_bin_locations(widths):
    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.0
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode='constant', value=0.0)
    return bin_locations


def search_inverse_bin(bin_left_cdf, inputs):
    return torchutils.searchsorted(bin_left_cdf, inputs)[..., None]


def search_forward_bin(bin_locations, inputs):
    return torchutils.searchsorted(bin_locations, inputs)[..., None]


def gather_input_values(bin_locations, widths, bin_left_cdf, bin_idx):
    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]
    return input_bin_locations, input_bin_widths, input_left_cdf


def gather_height_values(heights, bin_idx):
    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx + 1)[..., 0]
    return input_left_heights, input_right_heights


def calculate_quadratic_coefficients(input_left_heights, input_right_heights, input_bin_widths, input_left_cdf):
    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf
    return a, b, c


def calculate_outputs(inputs, inverse, a, b, c, input_bin_widths, input_left_heights, input_right_heights):
    if inverse:
        c_ = c - inputs
        alpha = (-b + torch.sqrt(b.pow(2) - 4 * a * c_)) / (2 * a)
        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights) + input_left_heights))
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log((alpha * (input_right_heights - input_left_heights) + input_left_heights))
    return outputs, logabsdet


def denormalize_outputs(outputs, inverse, left, right, bottom, top):
    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom
    return outputs