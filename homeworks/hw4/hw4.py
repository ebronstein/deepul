import argparse
import os

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
        self.downsample_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        # Downsample path
        # TODO: make this match pseudocode. Don't downsample at the last (deepest)
        # layer. There should be one downsample block per hidden_dim (except the)
        # last one, and there should be blocks_per_dim number of residual blocks
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
                self.downsample_layers.append(Downsample(prev_ch))
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
        # TODO: make this match pseudocode. Don't upsample at the first (deepest)
        # layer, and there should be an upsampling block per resblock (excluding
        # the first layer).
        # After the double for loop, self.up_blocks has len(hidden_dims) *
        # blocks_per_dim resblocks.
        # self.upsample_layers has len(hidden_dims) - 1 layers.
        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(
                blocks_per_dim + 1
            ):  # +1 for the additional block in upsampling
                dch = down_block_chans.pop()
                # NOTE: prev_ch + dch is equivalent to prev_ch * 2.
                self.up_blocks.append(
                    ResidualBlock(prev_ch + dch, hidden_dim, self.temb_channels)
                )
                prev_ch = hidden_dim
                # Only append an upsampling layer if we're not at the deepest
                # layer and we have added all the resblocks for that layer.
                if i != 0 and j == blocks_per_dim:
                    self.upsample_layers.append(Upsample(prev_ch))

        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=prev_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(prev_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        temb = timestep_embedding(t, self.hidden_dims[0])
        temb = self.emb_proj(temb)

        h = self.initial_conv(x)
        hs = [h]

        # Downsample
        for i in range(len(self.hidden_dims)):
            for j in range(self.blocks_per_dim):
                h = self.down_blocks[i * self.blocks_per_dim + j](h, temb)
                hs.append(h)
            if i != len(self.hidden_dims) - 1:
                h = self.downsample_layers[i](h)
                hs.append(h)

        # Bottleneck
        for block in self.mid_blocks:
            h = block(h, temb)

        # Upsample
        # print("Upsampling")
        for i, hidden_dim in list(enumerate(self.hidden_dims))[::-1]:
            for j in range(self.blocks_per_dim + 1):
                # print(f"i={i}, j={j}")
                skip_connection = hs.pop()
                # print("h:", h.shape)
                # print("skip_connection:", skip_connection.shape)
                h = torch.cat([h, skip_connection], dim=1)
                # print("h:", h.shape)
                up_block_index = -(i + 1) * (self.blocks_per_dim + 1) + j
                # print("up_block_index:", up_block_index)
                h = self.up_blocks[up_block_index](h, temb)
                # print("h:", h.shape)
                if i != 0 and j == self.blocks_per_dim:
                    # Negate the index because we're going backwards.
                    h = self.upsample_layers[-i](h)
                    # print("i != 0 and j == self.blocks_per_dim")
                    # print("h:", h.shape)
                # print()

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
        train_data,
        test_data,
        model=None,
        batch_size=1024,
        n_epochs=100,
        n_warmup_steps=100,
    ):
        self.n_epochs = n_epochs
        input_dim = train_data.shape[1]
        self.model = model or MLP(input_dim, input_dim)
        self.model = self.model.to(ptu.device)

        # Data loaders
        self.train_loader, self.test_loader = self.create_loaders(
            train_data, test_data, batch_size
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # LR scheduler
        n_iters_per_epoch = len(self.train_loader)
        n_iters = n_epochs * n_iters_per_epoch
        self.scheduler = warmup_cosine_decay_scheduler(
            self.optimizer, n_warmup_steps, n_iters
        )

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

    def compute_loss(self, x):
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
        eps_hat = self.model(x_t, t)

        # Step 5: Optimize the loss
        loss = (epsilon - eps_hat).pow(2).sum(axis=1).mean()
        return loss

    def eval(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(ptu.device)
                loss = self.compute_loss(x)
                total_loss += loss.item() * x.shape[0]

        return total_loss / len(test_loader.dataset)

    def train(self, log_freq=100, save_freq: int = 10, save_dir=None):
        train_losses = []
        test_losses = [self.eval(self.test_loader)]

        iter = 0
        for epoch in range(self.n_epochs):
            epoch_train_losses = []
            self.model.train()

            for x in self.train_loader:
                x = x.to(ptu.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(x)
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

        return train_losses, test_losses

    def ddpm_update(self, x, eps_hat, alpha_t, alpha_tm1, sigma_t, sigma_tm1):
        eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t.pow(2) / alpha_tm1.pow(2))
        update_term = alpha_tm1 * (x - sigma_t * eps_hat) / alpha_t
        noise_term = (
            torch.sqrt(torch.clamp(sigma_tm1.pow(2) - eta_t.pow(2), min=0)) * eps_hat
        )
        random_noise = eta_t * torch.randn_like(x, device=ptu.device)
        x_tm1 = update_term + noise_term + random_noise
        return x_tm1


def sample(model, num_samples, num_steps, return_steps, data_shape):
    model.model.eval()
    # Original: ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)
    ts = np.linspace(1 - 1e-4, 0, num_steps + 1)
    if not isinstance(data_shape, (list, tuple)):
        data_shape = (data_shape,)
    x_shape = (num_samples,) + tuple(data_shape)
    x = torch.randn(x_shape, device=ptu.device)
    exp_shape = [num_samples] + [1] * len(data_shape)

    samples = []
    for i in range(num_steps):
        t = ptu.tensor([ts[i]], dtype=torch.float32)
        tm1 = ptu.tensor([ts[i + 1]], dtype=torch.float32)

        alpha_t = model.get_alpha(t).expand(exp_shape)
        alpha_tm1 = model.get_alpha(tm1).expand(exp_shape)
        sigma_t = model.get_sigma(t).expand(exp_shape)
        sigma_tm1 = model.get_sigma(tm1).expand(exp_shape)

        eps_hat = model.model(x, t.expand(num_samples))
        x = model.ddpm_update(x, eps_hat, alpha_t, alpha_tm1, sigma_t, sigma_tm1)

        if i + 1 in return_steps:
            samples.append(x.cpu().detach().numpy())

    samples = np.array(samples)
    return samples


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
    train_losses, test_losses = model.train()

    return_steps = np.logspace(0, np.log10(512), num=9).astype(int)
    all_samples = sample(model, 2000, 512, return_steps, train_data.shape[1])
    all_samples = unnorm_mean_std(all_samples, mean, std)

    return train_losses, test_losses, all_samples


def q2(train_data, test_data):
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

    save_dir = "/nas/ucb/ebronstein/deepul/deepul/homeworks/hw4/models/q2"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses, test_losses = dmodel.train(save_dir=save_dir)

    return_steps = np.logspace(0, np.log10(512), num=10).astype(int)
    all_samples = []
    for _ in range(5):
        samples = sample(dmodel, 2, 512, return_steps, train_data.shape[1:])
        samples = samples.transpose(0, 1, 3, 4, 2)
        all_samples.append(samples)

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = unnorm_img(all_samples)

    return train_losses, test_losses, all_samples


if __name__ == "__main__":
    ptu.set_gpu_mode(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("question", choices=[1, 2], type=int)
    parser.add_argument("--part", choices=["a", "b"], required=False)

    args = parser.parse_args()

    if args.question == 1:
        q1_save_results(q1)
    elif args.question == 2:
        q2_save_results(q2)
    else:
        raise ValueError(
            f"Invalid question {args.question} and part {args.part} combination."
        )
