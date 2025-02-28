import torch
from torch import Tensor
from typing import NamedTuple
from torch.nn.utils.rnn import pad_sequence


class Batch(NamedTuple):
    r"""A single sample of the data."""

    # x_time: Tensor  # B×N:   the input timestamps.
    # x_vals: Tensor  # B×N×D: the input values.
    # x_mask: Tensor  # B×N×D: the input mask.
    #
    # y_time: Tensor  # B×K:   the target timestamps.
    # y_vals: Tensor  # B×K×D: the target values.
    # y_mask: Tensor  # B×K×D: teh target mask.

    x: Tensor       # B x T x (2D + 1) => val mask time
    target: Tensor  # B x K x (2D + 1)


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor


def tsdm_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_time: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

        context_x.append(t)
        context_vals.append(x)
        context_mask.append(mask_x)

        target_time.append(t_target)
        target_vals.append(y)
        target_mask.append(mask_y)

        # context_x.append(torch.cat([t, t_target], dim = 0))
        # x_vals_temp = torch.zeros_like(x)
        # y_vals_temp = torch.zeros_like(y)
        # context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        # context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # # context_y = torch.cat([context_vals, context_mask], dim=2)
        #
        # target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        # target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        # x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        # x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        # x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        # # y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        # y_time=pad_sequence(target_time, batch_first=True).squeeze(),
        # y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        # y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
        x=torch.cat([
                pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
                pad_sequence(context_mask, batch_first=True).squeeze(),
                pad_sequence(context_x, batch_first=True).squeeze().unsqueeze(-1),
            ], dim=-1),
        target=torch.cat([
                pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
                pad_sequence(target_mask, batch_first=True).squeeze(),
                pad_sequence(target_time, batch_first=True).squeeze().unsqueeze(-1),
            ], dim=-1),
    )
