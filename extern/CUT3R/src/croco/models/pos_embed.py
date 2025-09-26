# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


import numpy as np

import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token > 0:
        pos_embed = np.concatenate(
            [np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# ----------------------------------------------------------

# Prefer CUDA extension when available; otherwise define a PyTorch fallback so imports never fail.
try:
    from models.curope import cuRoPE2D  # type: ignore
    RoPE2D = cuRoPE2D  # noqa: N816 (keep class name as used elsewhere)
    print("Using CUDA-compiled version of RoPE2D")
except ImportError:
    print(
        "Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead"
    )

    class RoPE2D(torch.nn.Module):  # noqa: N801 (match expected class name)
        def __init__(self, freq: float = 100.0, F0: float = 1.0) -> None:
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D: int, seq_len: int, device, dtype):
            key = (D, seq_len, device, dtype)
            if key not in self.cache:
                inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, D, 2).float().to(device) / D)
                )
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[key] = (cos, sin)
            return self.cache[key]

        @staticmethod
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(
            self,
            tokens: torch.Tensor,
            pos1d: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> torch.Tensor:
            assert pos1d.ndim == 2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """
            tokens: [B, H, N, C]
            positions: [B, N, 2] (y, x)
            returns: [B, H, N, C]
            """
            assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim == 3 and positions.shape[-1] == 2
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
            y_half, x_half = tokens.chunk(2, dim=-1)
            y_half = self.apply_rope1d(y_half, positions[:, :, 0], cos, sin)
            x_half = self.apply_rope1d(x_half, positions[:, :, 1], cos, sin)
            return torch.cat((y_half, x_half), dim=-1)
