# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from ViT-Pytorch (https://github.com/lucidrains/vit-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vqvae import Codebook


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

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
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList(
                [
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                    PreNorm(dim, FeedForward(dim, mlp_dim)),
                ]
            )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: Union[Tuple[int, int], int],
        patch_size: Union[Tuple[int, int], int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        en_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.en_pos_embedding = nn.Parameter(
            torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)

        return x


class ViTDecoder(nn.Module):
    def __init__(
        self,
        image_size: Union[Tuple[int, int], int],
        patch_size: Union[Tuple[int, int], int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        de_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(
            torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False
        )
        self.to_pixel = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=image_height // patch_height),
            nn.ConvTranspose2d(
                dim, channels, kernel_size=patch_size, stride=patch_size
            ),
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        x = self.to_pixel(x)
        x = torch.tanh(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight


class BaseQuantizer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        straight_through: bool = True,
        use_norm: bool = True,
        use_residual: bool = False,
        num_quantizers: Optional[int] = None,
        embed_init: str = "normal",
    ) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        if embed_init == "normal":
            self.embedding.weight.data.normal_()
        elif embed_init == "uniform":
            self.embedding.weight.data.uniform_(-1 / self.n_embed, 1 / self.n_embed)
        else:
            raise ValueError(f"Unknown embedding initialization: {embed_init}")

    def quantize(
        self, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass

    def forward(
        self, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(
                partial(torch.stack, dim=-1), (losses, encoding_indices)
            )
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        beta: float = 0.25,
        use_norm: bool = True,
        use_residual: bool = False,
        num_quantizers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim, n_embed, True, use_norm, use_residual, num_quantizers
        )

        self.beta = beta

    def quantize(
        self, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # print("z.shape:", z.shape)
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        # print("z_reshaped_norm.shape:", z_reshaped_norm.shape)
        embedding_norm = self.norm(self.embedding.weight)

        d = (
            torch.sum(z_reshaped_norm**2, dim=1, keepdim=True)
            + torch.sum(embedding_norm**2, dim=1)
            - 2 * torch.einsum("b d, n d -> b n", z_reshaped_norm, embedding_norm)
        )

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # print("encoding_indices.shape:", encoding_indices.shape)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        # print("encoding_indices.shape:", encoding_indices.shape)

        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm) ** 2) + torch.mean(
            (z_qnorm - z_norm.detach()) ** 2
        )

        return z_qnorm, loss, encoding_indices


class Codebook(nn.Module):

    def __init__(self, codebook_size, code_dim):
        super().__init__()
        self.code_dim = code_dim
        self.codebook_size = codebook_size

        self.embedding = nn.Embedding(codebook_size, code_dim)
        # Initialize embedding to uniform random between -1/codebook_size and 1/codebook_size
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        # print("z.shape:", z.shape)
        flattened_z = z.view(-1, self.code_dim)
        # print("flattened_z.shape:", flattened_z.shape)
        weight = self.embedding.weight
        # Compute distances between z and embedding vectors
        distances = (
            (flattened_z**2).sum(dim=1, keepdim=True)
            - 2 * torch.mm(flattened_z, weight.t())
            + (weight.t() ** 2).sum(dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        # print("encoding_indices.shape:", encoding_indices.shape)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        code = self.embedding(encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((code.detach() - z) ** 2) + torch.mean(
            (code - z.detach()) ** 2
        )

        return code, loss, encoding_indices


class ViTVQ(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        encoder_dim: int = 256,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        encoder_mlp_dim: int = 1024,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        decoder_mlp_dim: int = 1024,
        code_dim: int = 256,
        code_size: int = 1024,
        quantizer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_dim=encoder_mlp_dim,
        )
        self.decoder = ViTDecoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            mlp_dim=decoder_mlp_dim,
        )
        quantizer_kwargs = quantizer_kwargs or {}
        self.quantizer = VectorQuantizer(code_dim, code_size, **quantizer_kwargs)
        # self.quantizer = Codebook(code_size, code_dim)
        self.pre_quant = nn.Linear(encoder_dim, code_dim)
        self.post_quant = nn.Linear(code_dim, decoder_dim)

    # def reconstruct(self, x):
    #     print("x.shape: ", x.shape)
    #     # Encode
    #     z = self.encoder(x)
    #     print("z.shape: ", z.shape)
    #     z = self.pre_quant(z)
    #     print("z.shape: ", z.shape)
    #     # Get the code
    #     code, code_stop_grad, _ = self.quantizer(z)
    #     # Decode the code token
    #     x_recon = self.decoder(self.post_quant(code_stop_grad))
    #     return x_recon

    # def forward(self, x):
    #     print("x.shape: ", x.shape)
    #     # Encode
    #     z = self.encoder(x)
    #     print("z.shape: ", z.shape)
    #     z = self.pre_quant(z)
    #     print("z.shape: ", z.shape)
    #     # Get the code
    #     code, code_stop_grad, _ = self.quantizer(z)
    #     # Decode the code token
    #     x_recon = self.decoder(self.post_quant(code_stop_grad))

    #     # Commitment loss
    #     commitment_loss = torch.mean((z - code.detach()) ** 2)
    #     # Embedding loss
    #     embedding_loss = torch.mean((code - z.detach()) ** 2)
    #     # Total regularization loss
    #     reg_loss = commitment_loss + embedding_loss

    #     return x_recon, reg_loss

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        quant, diff = self.encode(x)
        dec = self.decode(quant)

        return dec, diff

    def encode(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)

        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)

        return dec

    # def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
    #     h = self.encoder(x)
    #     h = self.pre_quant(h)
    #     _, _, codes = self.quantizer(h)

    #     return codes

    # def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
    #     quant = self.quantizer.embedding(code)
    #     quant = self.quantizer.norm(quant)

    #     if self.quantizer.use_residual:
    #         quant = quant.sum(-2)

    #     dec = self.decode(quant)

    #     return dec
