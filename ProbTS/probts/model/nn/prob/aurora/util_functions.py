from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x_tensor, new_shape):
    return F.interpolate(x_tensor.unsqueeze(0), size=new_shape, mode='linear').squeeze(0)


def resample(old: torch.Tensor, new_patch_len: int):
    assert old.dim() == 2, "the size of input tensor should be (d_model, patch_size)"
    if old.size(1) == new_patch_len:
        return old

    old = old.T
    old_shape = old.size(0)
    factor = new_patch_len / old_shape

    basis_vectors = torch.eye(old_shape, dtype=torch.get_default_dtype(), device=old.device)
    resize_mat = resize(basis_vectors, new_patch_len).T
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)

    resampled_kernels = resize_mat_pinv @ old * math.sqrt(factor)

    return resampled_kernels.T


def RoPE(query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to the query and key tensors.

    Args:
        query (torch.Tensor): Query tensor with shape (bs, head, max_len, output_dim).
        key (torch.Tensor): Key tensor with shape (bs, head, max_len, output_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensors after applying RoPE.
    """
    # Get the shape information of the input tensors
    batch_size, num_heads, max_len, output_dim = query.shape
    # Generate sinusoidal position embeddings
    pos_emb = sinusoidal_position_embedding(batch_size, num_heads, max_len, output_dim, query.device, factor=1)

    # Extract cosine and sine position embeddings
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # Apply RoPE to the query tensor
    query_rot = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1).reshape(query.shape)
    query = query * cos_pos + query_rot * sin_pos

    # Apply RoPE to the key tensor
    key_rot = torch.stack([-key[..., 1::2], key[..., ::2]], dim=-1).reshape(key.shape)
    key = key * cos_pos + key_rot * sin_pos

    return query, key


def RoPE_decoder(query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to the query and key tensors in the decoder.

    Args:
        query (torch.Tensor): Query tensor with shape (bs, head, q_max_len, output_dim).
        key (torch.Tensor): Key tensor with shape (bs, head, k_max_len, output_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensors after applying RoPE.
    """
    # Get the shape information of the input tensors
    batch_size, num_heads, q_max_len, output_dim = query.shape
    _, _, k_max_len, _ = key.shape
    # Generate sinusoidal position embeddings
    pos_emb = sinusoidal_position_embedding(batch_size, num_heads, k_max_len + q_max_len, output_dim, query.device,
                                            factor=1)

    # Extract cosine and sine position embeddings
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # Apply RoPE to the query tensor
    query_rot = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1).reshape(query.shape)
    query = query * cos_pos[:, :, -q_max_len:, :] + query_rot * sin_pos[:, :, -q_max_len:, :]

    # Apply RoPE to the key tensor
    key_rot = torch.stack([-key[..., 1::2], key[..., ::2]], dim=-1).reshape(key.shape)
    key = key * cos_pos[:, :, :k_max_len, :] + key_rot * sin_pos[:, :, :k_max_len, :]

    return query, key


def sinusoidal_position_embedding(
        batch_size: int,
        num_heads: int,
        max_len: int,
        output_dim: int,
        device: torch.device,
        factor: float = 1.0
) -> torch.Tensor:
    """
    Generate sinusoidal position embeddings.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.
        output_dim (int): Output dimension.
        device (torch.device): Device type.
        factor (float, optional): Scaling factor. Defaults to 1.0.

    Returns:
        torch.Tensor: Sinusoidal position embedding tensor with shape (bs, head, max_len, output_dim).
    """
    # Generate position indices
    position = torch.arange(0, max_len * factor, 1 / factor, dtype=torch.float).unsqueeze(-1)
    # Generate frequency indices
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)
    theta = torch.pow(10000, -2 * ids / output_dim)

    # Calculate position embeddings
    embeddings = position * theta
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # Expand dimensions to match batch size and number of attention heads
    embeddings = embeddings.repeat((batch_size, num_heads, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, num_heads, -1, output_dim))
    embeddings = embeddings.to(device)

    # If the factor is greater than 1, perform interpolation
    if factor > 1.0:
        interpolation_indices = torch.linspace(0, embeddings.shape[2] - 1, max_len).long()
        embeddings = embeddings[:, :, interpolation_indices, :]

    return embeddings


def causal_attention_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
