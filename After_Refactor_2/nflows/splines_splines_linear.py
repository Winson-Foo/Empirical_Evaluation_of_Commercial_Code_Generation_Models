import torch
from torch.nn import functional as F

def unconstrained_linear_spline(inputs, unnormalized_pdf, inverse=False, tail_bound=1.0, tails="linear"):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0

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


def linear_spline(inputs, unnormalized_pdf, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0):
    """
    Apply a linear spline transformation to the inputs.

    Args:
        inputs: The input tensor.
        unnormalized_pdf: The unnormalized probability density function tensor.
        inverse: Boolean value indicating whether to apply the inverse transformation.
        left: The left boundary of the spline.
        right: The right boundary of the spline.
        bottom: The bottom boundary of the spline.
        top: The top boundary of the spline.

    Returns:
        The transformed tensor and the log absolute determinant of the Jacobian.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    if inverse:
        inputs = normalize_inputs(inputs, bottom, top)
    else:
        inputs = normalize_inputs(inputs, left, right)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.0
    cdf = F.pad(cdf, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        outputs, logabsdet = inverse_linear_spline(inputs, cdf, pdf, num_bins)
    else:
        outputs, logabsdet = forward_linear_spline(inputs, cdf, pdf, num_bins)

    if inverse:
        outputs = denormalize_outputs(outputs, left, right)
    else:
        outputs = denormalize_outputs(outputs, bottom, top)

    return outputs, logabsdet


def normalize_inputs(inputs, left, right):
    """
    Normalize the inputs to the range [0, 1].

    Args:
        inputs: The input tensor.
        left: The left boundary of the range.
        right: The right boundary of the range.

    Returns:
        The normalized tensor.
    """
    return (inputs - left) / (right - left)


def denormalize_outputs(outputs, bottom, top):
    """
    Denormalize the outputs to the specified range.

    Args:
        outputs: The output tensor.
        bottom: The bottom boundary of the range.
        top: The top boundary of the range.

    Returns:
        The denormalized tensor.
    """
    return outputs * (top - bottom) + bottom


def inverse_linear_spline(inputs, cdf, pdf, num_bins):
    inv_bin_idx = torchutils.searchsorted(cdf, inputs)

    bin_boundaries = (
        torch.linspace(0, 1, num_bins + 1)
        .view([1] * inputs.dim() + [-1])
        .expand(*inputs.shape, -1)
    )

    slopes = (cdf[..., 1:] - cdf[..., :-1]) / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1])
    offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

    inv_bin_idx = inv_bin_idx.unsqueeze(-1)
    input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
    input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

    outputs = (inputs - input_offsets) / input_slopes
    outputs = torch.clamp(outputs, 0, 1)

    logabsdet = -torch.log(input_slopes)

    return outputs, logabsdet


def forward_linear_spline(inputs, cdf, pdf, num_bins):
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

    return outputs, logabsdet