import numpy as np
import torch
from torch.nn import functional as F
from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils


def unconstrained_linear_spline(inputs, unnormalized_pdf, inverse=False, tail_bound=1.0, tails="linear"):
    # Mask inputs outside the interval
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    # Initialize outputs and logabsdet tensors
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        # Handle inputs outside the interval
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        # Perform linear spline on inputs inside the interval
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = linear_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_pdf=unnormalized_pdf[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
        )

    return outputs, logabsdet


def linear_spline(inputs, unnormalized_pdf, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0):
    """
    Perform linear spline transformation on inputs.

    Reference: Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    # Check if inputs are within the specified domain
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    # Normalize inputs
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    # Get the number of bins
    num_bins = unnormalized_pdf.size(-1)

    # Softmax the unnormalized PDF
    pdf = F.softmax(unnormalized_pdf, dim=-1)

    # Compute the cumulative distribution function (CDF)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.0
    cdf = F.pad(cdf, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        # Inverse linear spline transformation
        inv_bin_idx = torchutils.searchsorted(cdf, inputs)

        bin_boundaries = torch.linspace(0, 1, num_bins + 1).view([1] * inputs.dim() + [-1]).expand(*inputs.shape, -1)

        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1])
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        # Forward linear spline transformation
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.float()

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    # Denormalize outputs
    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet