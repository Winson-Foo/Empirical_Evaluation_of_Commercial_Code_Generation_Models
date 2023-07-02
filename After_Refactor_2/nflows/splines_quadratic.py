import torch
from torch import Tensor
from torch.nn import functional as F
from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3


def unconstrained_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    inverse: bool = False,
    tail_bound: float = 1.0,
    tails: str = "linear",
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
) -> Tuple[Tensor, Tensor]:
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    num_bins = unnormalized_widths.shape[-1]

    if tails == "linear":
        outputs = get_linear_outputs(inputs, outside_interval_mask)
        logabsdet = torch.zeros_like(inputs)

        assert unnormalized_heights.shape[-1] == num_bins - 1
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        outputs_inside, logabsdet_inside = quadratic_spline(
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

        outputs[inside_interval_mask] = outputs_inside
        logabsdet[inside_interval_mask] = logabsdet_inside

    return outputs, logabsdet


def get_linear_outputs(inputs: Tensor, mask: Tensor) -> Tensor:
    outputs = torch.zeros_like(inputs)
    outputs[mask] = inputs[mask]
    return outputs


def quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
) -> Tuple[Tensor, Tensor]:
    check_input_bounds(inputs, left, right)

    if inverse:
        inputs = normalize_inputs(inputs, bottom, top, left, right)
    else:
        inputs = normalize_inputs(inputs, left, right, left, right)

    num_bins = unnormalized_widths.shape[-1]
    check_minimal_bin_width(min_bin_width, num_bins)
    check_minimal_bin_height(min_bin_height, num_bins)

    widths = compute_widths(unnormalized_widths, min_bin_width, num_bins)
    normalized_heights = compute_normalized_heights(
        unnormalized_heights, num_bins, min_bin_height
    )
    heights = normalize_heights(normalized_heights)

    bin_left_cdf = compute_bin_left_cdf(heights, widths)
    bin_locations = compute_bin_locations(widths)

    if inverse:
        bin_idx = torchutils.searchsorted(bin_left_cdf, inputs)[..., None]
    else:
        bin_idx = torchutils.searchsorted(bin_locations, inputs)[..., None]

    (
        input_bin_locations,
        input_bin_widths,
        input_left_cdf,
        input_left_heights,
        input_right_heights,
    ) = gather_bin_values(bin_idx, bin_locations, widths, heights)

    a, b, c = compute_abc_values(input_bin_widths, input_left_heights, input_left_cdf)

    if inverse:
        outputs, logabsdet = compute_inverse_outputs(
            inputs,
            a,
            b,
            c,
            right,
            left,
            input_bin_widths,
            input_right_heights,
            input_left_heights,
        )
    else:
        outputs, logabsdet = compute_forward_outputs(
            inputs,
            a,
            b,
            c,
            input_bin_widths,
            input_right_heights,
            input_left_heights,
        )

    if inverse:
        outputs = denormalize_outputs(outputs, right, left, bottom, top)
    else:
        outputs = denormalize_outputs(outputs, right, left, bottom, top)

    return outputs, logabsdet


def check_input_bounds(inputs: Tensor, left: float, right: float) -> None:
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()


def normalize_inputs(
    inputs: Tensor, in_bottom: float, in_top: float, out_bottom: float, out_top: float
) -> Tensor:
    return (inputs - in_bottom) / (in_top - in_bottom)


def compute_widths(
    unnormalized_widths: Tensor, min_bin_width: float, num_bins: int
) -> Tensor:
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    return widths


def check_minimal_bin_width(min_bin_width: float, num_bins: int) -> None:
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")


def check_minimal_bin_height(min_bin_height: float, num_bins: int) -> None:
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")


def compute_normalized_heights(
    unnormalized_heights: Tensor, num_bins: int, min_bin_height: float
) -> Tensor:
    unnorm_heights_exp = F.softplus(unnormalized_heights) + 1e-3
    if unnorm_heights_exp.shape[-1] == num_bins - 1:
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
        normalized_heights = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)
    return normalized_heights


def normalize_heights(normalized_heights: Tensor) -> Tensor:
    unnormalized_area = torch.sum(
        ((normalized_heights[..., :-1] + normalized_heights[..., 1:]) / 2) * widths,
        dim=-1,
    )[..., None]
    heights = normalized_heights / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights
    return heights


def compute_bin_left_cdf(heights: Tensor, widths: Tensor) -> Tensor:
    bin_left_cdf = torch.cumsum(
        ((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1
    )
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode="constant", value=0.0)
    return bin_left_cdf


def compute_bin_locations(widths: Tensor) -> Tensor:
    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.0
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode="constant", value=0.0)
    return bin_locations


def gather_bin_values(
    bin_idx: Tensor,
    bin_locations: Tensor,
    widths: Tensor,
    heights: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]
    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx + 1)[..., 0]
    return (
        input_bin_locations,
        input_bin_widths,
        input_left_cdf,
        input_left_heights,
        input_right_heights,
    )


def compute_abc_values(
    input_bin_widths: Tensor,
    input_left_heights: Tensor,
    input_left_cdf: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf
    return a, b, c


def compute_inverse_outputs(
    inputs: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    right: float,
    left: float,
    input_bin_widths: Tensor,
    input_right_heights: Tensor,
    input_left_heights: Tensor,
) -> Tuple[Tensor, Tensor]:
    c_ = c - inputs
    alpha = (-b + torch.sqrt(b.pow(2) - 4 * a * c_)) / (2 * a)
    outputs = alpha * input_bin_widths + input_bin_locations
    outputs = torch.clamp(outputs, 0, 1)
    logabsdet = -torch.log(
        (alpha * (input_right_heights - input_left_heights) + input_left_heights)
    )
    return outputs, logabsdet


def compute_forward_outputs(
    inputs: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    input_bin_widths: Tensor,
    input_right_heights: Tensor,
    input_left_heights: Tensor,
) -> Tuple[Tensor, Tensor]:
    alpha = (inputs - input_bin_locations) / input_bin_widths
    outputs = a * alpha.pow(2) + b * alpha + c
    outputs = torch.clamp(outputs, 0, 1)
    logabsdet = torch.log(
        (alpha * (input_right_heights - input_left_heights) + input_left_heights)
    )
    return outputs, logabsdet


def denormalize_outputs(
    outputs: Tensor, in_bottom: float, in_top: float, out_bottom: float, out_top: float
) -> Tensor:
    return outputs * (out_top - out_bottom) + out_bottom