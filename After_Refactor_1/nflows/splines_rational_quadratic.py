import numpy as np
import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    enable_identity_init: bool = False,
) -> torch.Tensor:
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
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = rational_quadratic_spline(
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
            enable_identity_init=enable_identity_init,
        )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    enable_identity_init: bool = False,
) -> torch.Tensor:
    _check_input_domain(inputs, left, right)
    num_bins = unnormalized_widths.shape[-1]
    _check_min_bin(min_bin_width, num_bins, "width")
    _check_min_bin(min_bin_height, num_bins, "height")

    widths = _compute_widths(unnormalized_widths, min_bin_width, num_bins, left, right)
    cumwidths = _compute_cumwidths(widths, left, right)
    widths = _compute_bin_widths(cumwidths)

    if enable_identity_init:
        beta = np.log(2) / (1 - min_derivative)
    else:
        beta = 1
    derivatives = _compute_derivatives(unnormalized_derivatives, min_derivative, beta)

    heights = _compute_heights(unnormalized_heights, min_bin_height, num_bins, bottom, top)
    cumheights = _compute_cumheights(heights, bottom, top)
    heights = _compute_bin_heights(cumheights)

    if inverse:
        bin_idx = torchutils.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = torchutils.searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = _get_input_cumwidths(cumwidths, bin_idx)
    input_bin_widths = _get_input_bin_widths(widths, bin_idx)

    input_cumheights = _get_input_cumheights(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _get_input_delta(delta, bin_idx)

    input_derivatives = _get_input_derivatives(derivatives, bin_idx)
    input_derivatives_plus_one = _get_input_derivatives_plus_one(derivatives, bin_idx)

    input_heights = _get_input_heights(heights, bin_idx)

    if inverse:
        outputs, logabsdet = _compute_spline_inverse(
            inputs, input_cumheights, input_derivatives, input_derivatives_plus_one,
            input_delta, input_heights, input_bin_widths
        )
    else:
        outputs, logabsdet = _compute_spline_forward(
            inputs, input_cumwidths, input_bin_widths,
            input_cumheights, input_delta, input_derivatives, input_derivatives_plus_one,
            input_heights,
        )

    return outputs, logabsdet


def _check_input_domain(inputs: torch.Tensor, left: float, right: float):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()


def _check_min_bin(min_bin_value: float, num_bins: int, bin_type: str):
    if min_bin_value * num_bins > 1.0:
        raise ValueError(f"Minimal bin {bin_type} too large for the number of bins")


def _compute_widths(unnormalized_widths: torch.Tensor, min_bin_width: float, num_bins: int,
                    left: float, right: float) -> torch.Tensor:
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    return widths


def _compute_cumwidths(widths: torch.Tensor, left: float, right: float) -> torch.Tensor:
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    return cumwidths


def _compute_bin_widths(cumwidths: torch.Tensor) -> torch.Tensor:
    return cumwidths[..., 1:] - cumwidths[..., :-1]


def _compute_derivatives(unnormalized_derivatives: torch.Tensor, min_derivative: float, beta: float) -> torch.Tensor:
    derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=beta)
    return derivatives


def _compute_heights(unnormalized_heights: torch.Tensor, min_bin_height: float, num_bins: int,
                     bottom: float, top: float) -> torch.Tensor:
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    return heights


def _compute_cumheights(heights: torch.Tensor, bottom: float, top: float) -> torch.Tensor:
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    return cumheights


def _compute_bin_heights(cumheights: torch.Tensor) -> torch.Tensor:
    return cumheights[..., 1:] - cumheights[..., :-1]


def _get_input_cumwidths(cumwidths: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return cumwidths.gather(-1, bin_idx)[..., 0]


def _get_input_bin_widths(widths: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return widths.gather(-1, bin_idx)[..., 0]


def _get_input_cumheights(cumheights: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return cumheights.gather(-1, bin_idx)[..., 0]


def _get_input_delta(delta: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return delta.gather(-1, bin_idx)[..., 0]


def _get_input_derivatives(derivatives: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return derivatives.gather(-1, bin_idx)[..., 0]


def _get_input_derivatives_plus_one(derivatives: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return derivatives[..., 1:].gather(-1, bin_idx)[..., 0]


def _get_input_heights(heights: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
    return heights.gather(-1, bin_idx)[..., 0]


def _compute_spline_inverse(
    inputs: torch.Tensor,
    input_cumheights: torch.Tensor,
    input_derivatives: torch.Tensor,
    input_derivatives_plus_one: torch.Tensor,
    input_delta: torch.Tensor,
    input_heights: torch.Tensor,
    input_bin_widths: torch.Tensor
) -> torch.Tensor:
    a = (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) + input_heights * (input_delta - input_derivatives)
    b = input_heights * input_derivatives - (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    )
    c = -input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    derivative_numerator = input_delta.pow(2)
