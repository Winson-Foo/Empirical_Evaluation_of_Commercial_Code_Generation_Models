import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils


def unconstrained_linear_spline(
    inputs: torch.Tensor,
    unnormalized_pdf: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 1.0,
    tails: str = "linear"
) -> torch.Tensor:
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
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


def linear_spline(
    inputs: torch.Tensor,
    unnormalized_pdf: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0
) -> torch.Tensor:
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.0
    cdf = F.pad(cdf, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        inv_bin_idx = torchutils.searchsorted(cdf, inputs)

        bin_boundaries = (
            torch.linspace(0, 1, num_bins + 1)
            .view([1] * inputs.dim() + [-1])
            .expand(*inputs.shape, -1)
        )

        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (
            bin_boundaries[..., 1:] - bin_boundaries[..., :-1]
        )
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
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

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet