# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import functools

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


@functools.lru_cache(maxsize=1)
def _inplace_conv_state_enabled() -> bool:
    from vllm_ascend.ascend_config import get_ascend_config

    return bool(get_ascend_config().enable_kda_conv_state_inplace)


def _stride_writable(conv_state: torch.Tensor) -> bool:
    """Whether aclnn can mutate this state view in place.

    Dimension-first (DS) layouts are rejected by aclnn (561002) and negative
    strides cannot be addressed, so both keep the staging path.
    """
    return conv_state.stride(2) == 1 and conv_state.stride(0) > 0 and conv_state.stride(1) > 0


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    run_mode: int,
    initial_state_mode: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state."""
    # Padded requests are skipped by the kernel, so their output rows are only
    # "clean" because they are pre-zeroed. Decode never reads them: every
    # stateful write (conv_state, recurrent_state, KV cache) skips PAD_SLOT_ID
    # and sampling indexes real requests only. The fast path therefore skips
    # the 34 per-step zeroing launches; prefill keeps the zeroed buffer.
    if run_mode == 1 and _inplace_conv_state_enabled():
        output = torch.empty_like(x)
    else:
        output = torch.zeros_like(x)
    if cache_indices.shape[0] == 0:
        return output
    kernel_state = conv_state
    kernel_indices = cache_indices
    null_block_id = 0
    # aclnnCausalConv1d materializes a non-contiguous state without writing its
    # mutations back to the view. Stage only this batch's rows, retaining both
    # page strides and DS layouts; never copy the entire persistent cache.
    staged = not conv_state.is_contiguous()
    # Decode-side UPDATE can mutate the strided cache in place, which removes
    # the pack/writeback copies entirely. Prefill keeps staging: the in-place
    # contract is only validated for UPDATE. Padding rows reach the kernel as
    # NULL_BLOCK_ID, folded by the metadata builder once per step, because
    # UPDATE reads out of bounds on negative indices.
    if staged and run_mode == 1 and _inplace_conv_state_enabled() and _stride_writable(conv_state):
        # null_block_id stays 0 = NULL_BLOCK_ID, the value padding rows carry.
        staged = False
        if cache_indices.dim() == 2:
            kernel_indices = cache_indices[:, 0]
    if staged:
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        # UPDATE does not recognize PAD_SLOT_ID: a negative null_block_id
        # disables its null-block check. Reserve a non-negative slot after
        # the request rows, keeping packed request zero a valid state.
        null_block_id = requests
        kernel_state = torch.empty((requests + 1, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
        kernel_indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device)
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
    kernel_indices = kernel_indices.contiguous()
    if run_mode == 0:
        result = causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states=kernel_state,
            query_start_loc=query_start_loc,
            cache_indices=kernel_indices,
            has_initial_state=initial_state_mode,
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
            null_block_id=null_block_id,
        )
    else:
        result = causal_conv1d_update(
            x,
            kernel_state,
            weight,
            bias=None,
            activation="silu",
            conv_state_indices=kernel_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            null_block_id=null_block_id,
            out=output,
        )
    if staged:
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result
