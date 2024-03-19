import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deepul import pytorch_util as ptu
from deepul.hw4_helper import *


def norm_mean_std(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return train_data, test_data, mean, std


def unnorm_mean_std(data, mean, std):
    return data * std + mean


def norm_img(data):
    return data * 2 - 1


def unnorm_img(data):
    return (data + 1) / 2


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            # +1 for time concatenation
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_t, t):
        # x_t: (batch_size, input_dim)
        # t: (batch_size,)
        t = t.view(-1, 1)  # Reshape t to (batch_size, 1)
        x = torch.cat((x_t, t), dim=-1)  # Concatenate x and t
        return self.net(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(
        -np.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
        / half_dim
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.proj = None

    def forward(self, x, temb):
        h = F.silu(self.gn1(self.conv1(x)))
        temb = self.temb_proj(temb)
        h += temb[:, :, None, None]
        h = F.silu(self.gn2(self.conv2(h)))
        if self.proj:
            x = self.proj(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim):
        super(UNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.blocks_per_dim = blocks_per_dim
        self.temb_channels = hidden_dims[0] * 4
        self.emb_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], self.temb_channels),
            nn.SiLU(),
            nn.Linear(self.temb_channels, self.temb_channels),
        )
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Downsample path
        # There should be one downsample block per hidden_dim except the last
        # one, and there should be blocks_per_dim number of residual blocks
        # per layer.
        prev_ch = hidden_dims[0]
        # After the double for loop, down_block_chans = [hidden_dims[0]] flatten([[hd] * (blocks_per_dim + 1) for hd in hidden_dims[:-1]]) + [hidden_dims[-1]] * blocks_per_dim
        # self.down_blocks has len(hidden_dims) * blocks_per_dim resblocks.
        # self.downsample_layers has len(hidden_dims) - 1 layers.
        down_block_chans = [prev_ch]
        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(blocks_per_dim):
                self.down_blocks.append(
                    ResidualBlock(prev_ch, hidden_dim, self.temb_channels)
                )
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)
            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_ch))
                down_block_chans.append(prev_ch)

        # print("After downsample path:")
        # print("down_block_chans:", down_block_chans)

        # Bottleneck. prev_ch = hidden_dims[-1] here.
        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock(prev_ch, prev_ch, self.temb_channels),
                ResidualBlock(prev_ch, prev_ch, self.temb_channels),
            ]
        )

        # Upsample path
        # Don't upsample at the last (shallowest) layer, and there should be an
        # upsampling block per resblock excluding the last (shallowest) layer.
        # After the double for loop, self.up_blocks has len(hidden_dims) *
        # blocks_per_dim resblocks.
        # self.upsample_layers has len(hidden_dims) - 1 layers.
        # i, hidden_dim = (3, 512), (2, 256), (1, 128), (0, 64)
        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(
                blocks_per_dim + 1
            ):  # +1 for the additional block in upsampling
                dch = down_block_chans.pop()
                # print("Popped dch:", dch)
                # NOTE: prev_ch + dch is equivalent to prev_ch * 2.
                self.up_blocks.append(
                    ResidualBlock(prev_ch + dch, hidden_dim, self.temb_channels)
                )
                prev_ch = hidden_dim
                # Only append an upsampling layer if we're not at the deepest
                # layer and we have added all the resblocks for that layer.
                if i != 0 and j == blocks_per_dim:
                    self.up_blocks.append(Upsample(prev_ch))

        # print("After upsample path:")
        # print("down_block_chans:", down_block_chans)

        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=prev_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(prev_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        temb = timestep_embedding(t, self.hidden_dims[0])
        temb = self.emb_proj(temb)

        h = self.initial_conv(x)
        hs = [h]

        # Downsample
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, temb)
            else:
                h = block(h)

            hs.append(h)

        # print("After downsample path:")
        # print("len(hs):", len(hs))

        # Bottleneck
        for block in self.mid_blocks:
            h = block(h, temb)

        # Upsample
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
            else:
                h = block(h)

        # print("After upsample path:")
        # print("len(hs):", len(hs))

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        return h


def warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a scheduler with warmup followed by cosine decay.

    Args:
        optimizer: Optimizer linked to the model parameters.
        warmup_steps: Number of steps for the warmup phase.
        total_steps: Total number of steps in the training.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DiffusionModel(object):
    def __init__(
        self,
        train_data=None,
        test_data=None,
        model=None,
        batch_size=1024,
        n_epochs=100,
        n_warmup_steps=100,
        has_labels=False,
    ):
        self.n_epochs = n_epochs
        self.has_labels = has_labels

        # Data loaders
        if isinstance(train_data, torch.utils.data.DataLoader):
            assert isinstance(test_data, torch.utils.data.DataLoader)
            self.train_loader = train_data
            self.test_loader = test_data
            train_data_shape = None
        elif train_data is not None:
            assert test_data is not None
            train_data_shape = train_data.shape
            self.train_loader, self.test_loader = self.create_loaders(
                train_data, test_data, batch_size
            )
        else:
            self.train_loader = None
            self.test_loader = None

        if model is None:
            assert train_data_shape is not None and len(train_data_shape) == 2
            input_dim = train_data_shape[1]
            self.model = MLP(input_dim, input_dim)
        else:
            self.model = model
        self.model = self.model.to(ptu.device)

        def model_with_labels(x, labels, t, **kwargs):
            return self.model(x, labels, t, **kwargs)

        def model_without_labels(x, labels, t):
            return self.model(x, t)

        if has_labels:
            self.model_fn = model_with_labels
        else:
            self.model_fn = model_without_labels

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # LR scheduler
        if self.train_loader is not None:
            n_iters_per_epoch = len(self.train_loader)
            n_iters = n_epochs * n_iters_per_epoch
            self.scheduler = warmup_cosine_decay_scheduler(
                self.optimizer, n_warmup_steps, n_iters
            )
        else:
            self.scheduler = None

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def create_loaders(self, train_data, test_data, batch_size):
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def get_alpha(self, t):
        return torch.cos(np.pi / 2 * t).to(ptu.device)

    def get_sigma(self, t):
        return torch.sin(np.pi / 2 * t).to(ptu.device)

    def compute_loss(self, x, labels=None):
        batch_size = x.shape[0]

        # Step 1: Sample diffusion timestep uniformly in [0, 1]
        t = torch.rand(batch_size, device=ptu.device)  # [batch_size]

        # Step 2: Compute noise-strength
        alpha_t = self.get_alpha(t)
        sigma_t = self.get_sigma(t)

        # Step 3: Apply forward process
        epsilon = torch.randn_like(x, device=ptu.device)
        exp_shape = [batch_size] + [1] * (len(x.shape) - 1)
        # print(alpha_t.shape, x.shape, sigma_t.shape, epsilon.shape)
        # print("exp_shape", exp_shape)
        alpha_t = alpha_t.view(exp_shape)
        sigma_t = sigma_t.view(exp_shape)
        x_t = alpha_t * x + sigma_t * epsilon  # x.shape

        # Step 4: Estimate epsilon
        eps_hat = self.model_fn(x_t, labels, t)

        # Step 5: Optimize the loss
        # loss = (epsilon - eps_hat).pow(2).sum(axis=1).mean()
        loss = (epsilon - eps_hat).pow(2).mean()
        return loss

    def eval(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in test_loader:
                if self.has_labels:
                    x, labels = x
                    labels = labels.to(ptu.device)
                else:
                    labels = None
                x = x.to(ptu.device)
                loss = self.compute_loss(x, labels)
                total_loss += loss.item() * x.shape[0]

        return total_loss / len(test_loader.dataset)

    def train(self, log_freq=100, save_freq: int = 10, save_dir=None):
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses = []
        test_losses = [self.eval(self.test_loader)]

        iter = 0
        for epoch in range(self.n_epochs):
            epoch_train_losses = []
            self.model.train()

            for x in self.train_loader:
                if self.has_labels:
                    x, labels = x
                    labels = labels.to(ptu.device)
                else:
                    labels = None
                x = x.to(ptu.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(x, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_train_losses.append(loss.item())

                if log_freq is not None and iter % log_freq == 0:
                    print(f"Epoch {epoch+1}, iter {iter}, Loss: {loss.item()}")

                iter += 1

            train_losses.extend(epoch_train_losses)
            test_losses.append(self.eval(self.test_loader))

            if save_dir is not None and epoch % save_freq == 0:
                self.save(os.path.join(save_dir, f"diffusion_model_epoch_{epoch}.pt"))

        if save_dir is not None:
            self.save(os.path.join(save_dir, "diffusion_model_final.pt"))
            np.save(os.path.join(save_dir, "train_losses.npy"), train_losses)
            np.save(os.path.join(save_dir, "test_losses.npy"), test_losses)

        return train_losses, test_losses


def ddpm_update(x, eps_hat, alpha_t, alpha_tm1, sigma_t, sigma_tm1, clip=None):
    eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t.pow(2) / alpha_tm1.pow(2))
    x_tm1_mean = (x - sigma_t * eps_hat) / alpha_t
    if clip is not None:
        min, max = clip
        x_tm1_mean = torch.clamp(x_tm1_mean, min, max)
    update_term = alpha_tm1 * x_tm1_mean
    noise_term = (
        torch.sqrt(torch.clamp(sigma_tm1.pow(2) - eta_t.pow(2), min=0)) * eps_hat
    )
    random_noise = eta_t * torch.randn_like(x, device=ptu.device)
    x_tm1 = update_term + noise_term + random_noise
    return x_tm1


def sample(
    model,
    num_samples,
    return_steps,
    data_shape,
    labels=None,
    clip=None,
    cfg_w=None,
    null_class=None,
):
    model.model.eval()
    if not isinstance(data_shape, (list, tuple)):
        data_shape = (data_shape,)
    x_shape = (num_samples,) + tuple(data_shape)
    exp_shape = [num_samples] + [1] * len(data_shape)
    samples = []  # [num_labels, len(return_steps), num_samples, *data_shape]

    if cfg_w is not None:
        assert labels is not None
        assert null_class is not None
        with torch.no_grad():
            null_class = ptu.tensor(null_class, dtype=torch.int32).expand(num_samples)

    if labels is None:
        labels = [None]
        model_kwargs = {}
    else:
        model_kwargs = {"training": False}

    for label in labels:
        label_samples = []
        with torch.no_grad():
            if label is not None:
                label = ptu.tensor(label, dtype=torch.int32)
                label = label.expand(num_samples)
            for num_steps in return_steps:
                ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)

                x = torch.randn(x_shape, device=ptu.device)
                for i in range(num_steps):
                    t = ptu.tensor([ts[i]], dtype=torch.float32)
                    tm1 = ptu.tensor([ts[i + 1]], dtype=torch.float32)

                    alpha_t = model.get_alpha(t).expand(exp_shape)
                    alpha_tm1 = model.get_alpha(tm1).expand(exp_shape)
                    sigma_t = model.get_sigma(t).expand(exp_shape)
                    sigma_tm1 = model.get_sigma(tm1).expand(exp_shape)

                    eps_hat = model.model_fn(
                        x, label, t.expand(num_samples), **model_kwargs
                    )
                    if cfg_w is not None:
                        eps_hat_null = model.model_fn(
                            x, null_class, t.expand(num_samples), **model_kwargs
                        )
                        eps_hat = eps_hat_null + cfg_w * (eps_hat - eps_hat_null)

                    x = ddpm_update(
                        x, eps_hat, alpha_t, alpha_tm1, sigma_t, sigma_tm1, clip=clip
                    )

                label_samples.append(x.cpu().detach().numpy())
            samples.append(label_samples)

    # Squeeze out the label and return_steps dimensions if there's only.
    samples = np.array(samples)
    if len(labels) == 1:
        samples = samples.squeeze(0)
    return samples


def dropout_classes(y, null_class, dropout_prob=0.1):
    """Randomly dropout classes with a given probability."""
    dropout_mask = torch.rand(y.shape) < dropout_prob
    y[dropout_mask] = null_class
    return y


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def patchify_flatten(images, patch_size):
    """
    Convert encoded images to flattened patches.

    Args:
        images: Tensor of shape (B, D, H, W)
        patch_size: Size of each patch (P)

    Returns:
        Flattened patches of shape (B, L, patch_size * patch_size * D) where L
        is the number of patches.
    """
    B, D, H, W = images.shape
    # [B, D, H // patch_size, patch_size, W // patch_size, patch_size]
    images = images.reshape(
        B, D, H // patch_size, patch_size, W // patch_size, patch_size
    )
    # [B, H // patch_size, W // patch_size, patch_size, patch_size, D]
    images = images.permute(0, 2, 4, 3, 5, 1)
    # [B, H // patch_size * W // patch_size, patch_size, patch_size, D]
    images = images.flatten(1, 2)
    # [B, H // patch_size * W // patch_size, patch_size * patch_size * D]
    return images.flatten(2, 4)


def unpatchify(patches, patch_size, H, W, D):
    """
    Convert flattened patches back to encoded images.

    Args:
        patches: Tensor of shape (B, L, patch_size * patch_size * D)
        patch_size: Size of each patch (P)
        H: Original height of the image
        W: Original width of the image
        D: Number of latent dimensions in the encoded image

    Returns:
        Images of shape (B, D, H, W).
    """
    B = patches.shape[0]
    patches = patches.reshape(
        B, H // patch_size, W // patch_size, patch_size, patch_size, D
    )
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    images = patches.reshape(B, D, H, W)
    return images


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        """
        Args:
            images: The image data of shape (N, H, W, D)
            labels: The label data of shape (N,)
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label


class DiTMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        # TODO: maybe apply LayerNorm first.
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, ar_mask: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear layers that project the input to Q, K, and V for all heads.
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        if ar_mask:
            # Create a mask for autoregressive property.
            # 0 means valid position, 1 means masked position.
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            mask = mask.masked_fill(mask == 1, -1e9)
        else:
            mask = torch.zeros(seq_len, seq_len)
        self.register_buffer("mask", mask)

        self.cache = None

    def split_heads(self, x, batch_size):
        """Split x into self.num_heads pieces.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # Shape: [batch_size, seq_len, num_heads, depth]
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, use_cache=False):
        """Apply forward pass.

        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            v: Value tensor of shape (batch_size, seq_len, d_model)
            use_cache: Whether to use cache for fast decoding. If True, q, k, and v
                have shape (batch_size, 1, d_model).

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = q.size(0)

        # [batch_size, num_heads, 1, depth]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        if use_cache:
            if self.cache is not None:
                # Concatenate with the cached keys and values
                # [batch_size, num_heads, seq_len, depth]
                k = torch.cat([self.cache["k"], k], dim=2)
                v = torch.cat([self.cache["v"], v], dim=2)
            # Update cache
            self.cache = {"k": k, "v": v}

        # [batch_size, num_heads, seq_len, seq_len]
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / self.depth**0.5

        # Apply mask to the scaled attention logits if not using cache.
        if not use_cache:
            seq_len = q.size(2)
            scaled_attention_logits += self.mask[:seq_len, :seq_len]

        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Shape: [batch_size, num_heads, seq_len, depth]
        output = torch.matmul(attention_weights, v)
        # Concatenate the output of all heads
        output = (
            output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.dense(output)

    def clear_cache(self):
        self.cache = None

    def is_cache_empty(self):
        return self.cache is None


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.linear2 = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.linear1(c)
        shift, scale = c.chunk(2, dim=1)
        x = self.layer_norm(x)
        x = modulate(x, shift, scale)
        x = self.linear2(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, seq_len):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mha = MultiHeadAttention(hidden_size, num_heads, seq_len)
        self.mlp = DiTMLP(hidden_size)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(
            6, dim=1
        )

        # print("x:", x.shape)
        h = self.layer_norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.mha(h, h, h)

        h = self.layer_norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_shape,
        patch_size=2,
        hidden_size=512,
        num_heads=8,
        num_layers=12,
        num_classes=10,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        _, D, H, W = input_shape
        self.H, self.W, self.D = H, W, D
        L = patch_size**2 * D
        self.proj = nn.Linear(L, hidden_size)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, H // patch_size)
        pos_embed = ptu.tensor(pos_embed, dtype=torch.float32)
        self.register_buffer("pos_embed", pos_embed)

        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        seq_len = patch_size**2 * D
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, seq_len) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, D)
        self.dropout_prob = dropout_prob

    def forward(self, x, y, t, training=True):
        # B, D, H, W = x.shape
        # print("x:", x.shape)
        # [B, L = H // P * W // P, P*P*D] = [B, 16, 16]
        x = patchify_flatten(x, self.patch_size)
        # print("x:", x.shape)
        x = self.proj(x)  # [B, L, hidden_size]
        # print("x:", x.shape)
        x += self.pos_embed.unsqueeze(0)  # [B, L, hidden_size]
        # print("x:", x.shape)

        t_emb = timestep_embedding(t, self.hidden_size)
        if training:
            y = dropout_classes(y, self.num_classes, dropout_prob=self.dropout_prob)
        y_emb = self.embedding(y)
        c = t_emb + y_emb
        # print("c:", c.shape)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = unpatchify(x, self.patch_size, self.H, self.W, self.D)
        return x


def q1(train_data, test_data):
    """
    train_data: A (100000, 2) numpy array of 2D points
    test_data: A (10000, 2) numpy array of 2D points

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (9, 2000, 2) of samples drawn from your model.
      Draw 2000 samples for each of 9 different number of diffusion sampling steps
      of evenly logarithmically spaced integers 1 to 512
      hint: np.power(np.linspace(0, 9, 9), 2).astype(int)
    """
    train_data, test_data, mean, std = norm_mean_std(train_data, test_data)

    model = DiffusionModel(
        train_data, test_data, batch_size=1024, n_epochs=100, n_warmup_steps=100
    )

    # Load
    load_dir = (
        "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q1/2024-03-18_10-27-15"
    )
    model.load(os.path.join(load_dir, "diffusion_model_final.pt"))
    train_losses = np.load(os.path.join(load_dir, "train_losses.npy"))
    test_losses = np.load(os.path.join(load_dir, "test_losses.npy"))

    # Save
    # save_dir = "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q1"
    # train_losses, test_losses = model.train(save_dir=save_dir)

    return_steps = np.logspace(0, np.log10(512), num=9).astype(int)
    all_samples = sample(model, 2000, return_steps, train_data.shape[1:])
    all_samples = unnorm_mean_std(all_samples, mean, std)

    return train_losses, test_losses, all_samples


def q2(train_data, test_data, load=False):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(np.linspace(0, 9, 10), 2).astype(int)
    """
    train_data = train_data.transpose(0, 3, 1, 2)
    test_data = test_data.transpose(0, 3, 1, 2)

    # Normalize to [-1, 1]
    train_data = norm_img(train_data)
    test_data = norm_img(test_data)

    unet = UNet(3, [64, 128, 256, 512], 2)

    dmodel = DiffusionModel(
        train_data,
        test_data,
        model=unet,
        batch_size=256,
        n_epochs=60,
        n_warmup_steps=100,
    )

    if load:
        load_dir = "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q2/2024-03-18_16-32-33"
        dmodel.load(os.path.join(load_dir, "diffusion_model_final.pt"))
        train_losses = np.load(os.path.join(load_dir, "train_losses.npy"))
        test_losses = np.load(os.path.join(load_dir, "test_losses.npy"))
    else:
        save_dir = "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q2"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_losses, test_losses = dmodel.train(save_dir=save_dir)

    return_steps = np.logspace(0, np.log10(512), num=10).astype(int)
    clip = (-1, 1)
    samples = sample(dmodel, 10, return_steps, train_data.shape[1:], clip=clip)
    samples = samples.transpose(0, 1, 3, 4, 2)
    samples = unnorm_img(samples)

    return train_losses, test_losses, samples


def q3_b(train_data, train_labels, test_data, test_labels, vae):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    train_labels: A (50000,) numpy array of class labels
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]
    test_labels: A (10000,) numpy array of class labels
    vae: a pretrained VAE

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific class (i.e. row 0 is class 0, row 1 class 1, ...). Use 512 diffusion timesteps
    """
    train_data = norm_img(train_data)
    test_data = norm_img(test_data)
    train_data = ptu.tensor(train_data).float().permute(0, 3, 1, 2)
    test_data = ptu.tensor(test_data).float().permute(0, 3, 1, 2)

    # [50000, 4, 8, 8]
    enc_train_data = vae.encode(train_data)
    enc_test_data = vae.encode(test_data)

    scale_factor = 1.2716
    enc_train_data = enc_train_data / scale_factor
    enc_test_data = enc_test_data / scale_factor

    batch_size = 256
    train_dataset = CustomImageDataset(enc_train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = CustomImageDataset(enc_test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    patch_size = 2
    hidden_size = 512
    num_heads = 8
    num_layers = 12
    num_classes = 10
    dit = DiT(
        enc_train_data.shape,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    dmodel = DiffusionModel(
        train_loader,
        test_loader,
        model=dit,
        n_epochs=60,
        n_warmup_steps=100,
        has_labels=True,
    )

    save_dir = "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q3"
    train_losses, test_losses = dmodel.train(save_dir=save_dir)

    # Sample
    num_labels = 10
    num_samples = 10
    num_steps = 512
    clip = (-1, 1)
    enc_samples = sample(
        dmodel,
        num_samples,
        [num_steps],
        enc_train_data.shape[1:],
        labels=list(range(num_labels)),
        clip=clip,
    )
    # Select the only return step of 512.
    enc_samples = enc_samples[:, 0]  # [num_labels=10, num_samples=10, 4, 8, 8]
    flat_enc_samples = enc_samples.reshape(
        -1, 4, 8, 8
    )  # [num_labels * num_samples = 100, 4, 8, 8]
    # Decode
    flat_enc_samples *= scale_factor
    # [num_labels * num_samples = 100, 3, 32, 32]
    flat_samples = vae.decode(flat_enc_samples).detach().cpu().numpy()
    samples = flat_samples.reshape(num_labels, num_samples, 3, 32, 32)
    samples = samples.transpose(0, 1, 3, 4, 2)
    samples = unnorm_img(samples)

    return train_losses, test_losses, samples


if __name__ == "__main__":
    ptu.set_gpu_mode(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("question", choices=[1, 2, 3], type=int)
    parser.add_argument("--part", choices=["a", "b"], required=False)

    args = parser.parse_args()

    if args.question == 1:
        q1_save_results(q1)
    elif args.question == 2:
        q2_save_results(q2)
    elif args.question == 3 and args.part == "b":
        q3b_save_results(q3_b)
    else:
        raise ValueError(
            f"Invalid question {args.question} and part {args.part} combination."
        )
