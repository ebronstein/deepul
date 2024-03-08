import argparse
import os
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim
import torch.utils.data as data
import torchvision
import vit
from einops import repeat
from scipy.stats import norm
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook, trange
from vit_pytorch import ViT

import deepul.pytorch_util as ptu
import deepul.utils as deepul_utils
from deepul.hw3_helper import *
from deepul.hw3_utils.lpips import LPIPS

warnings.filterwarnings("ignore")
ptu.set_gpu_mode(True)


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = (
            torch.stack(stack, 0)
            .transpose(0, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, s_height, s_width, s_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class Upsample_Conv2d(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True
    ):
        super(Upsample_Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        _x = torch.cat([x, x, x, x], dim=1)
        _x = self.depth_to_space(_x)
        _x = self.conv(_x)
        return _x


def maybe_add_spectral_norm(module, add_spectral_norm=False):
    return spectral_norm(module) if add_spectral_norm else module


class Downsample_Conv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=True,
        add_spectral_norm=False,
    ):
        super(Downsample_Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.conv = maybe_add_spectral_norm(
            self.conv, add_spectral_norm=add_spectral_norm
        )
        self.space_to_depth = SpaceToDepth(2)

    def forward(self, x):
        _x = self.space_to_depth(x)
        _x = sum(_x.chunk(4, dim=1)) / 4.0
        _x = self.conv(_x)
        return _x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super(ResnetBlockUp, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.BatchNorm2d(in_dim),
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1),
                Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0),
            ]
        )

    def forward(self, x):
        _x = x
        for i in range(len(self.layers) - 1):
            _x = self.layers[i](_x)
        return self.layers[-1](x) + _x


class ResnetBlockDown(nn.Module):
    def __init__(
        self,
        in_dim,
        kernel_size=(3, 3),
        stride=1,
        n_filters=256,
        add_spectral_norm=False,
        leaky_relu=False,
    ):
        super(ResnetBlockDown, self).__init__()
        relu_cls = nn.LeakyReLU if leaky_relu else nn.ReLU
        self.layers = nn.ModuleList(
            [
                relu_cls(),
                maybe_add_spectral_norm(
                    nn.Conv2d(in_dim, n_filters, kernel_size, stride=stride, padding=1),
                    add_spectral_norm=add_spectral_norm,
                ),
                relu_cls(),
                Downsample_Conv2d(
                    n_filters,
                    n_filters,
                    kernel_size,
                    add_spectral_norm=add_spectral_norm,
                ),
                Downsample_Conv2d(
                    in_dim,
                    n_filters,
                    kernel_size=(1, 1),
                    padding=0,
                    add_spectral_norm=add_spectral_norm,
                ),
            ]
        )

    def forward(self, x):
        _x = x
        for i in range(len(self.layers) - 1):
            _x = self.layers[i](_x)
        return self.layers[-1](x) + _x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        kernel_size=(3, 3),
        n_filters=256,
        add_spectral_norm=False,
        leaky_relu=False,
    ):
        super(ResBlock, self).__init__()
        relu_cls = nn.LeakyReLU if leaky_relu else nn.ReLU
        self.layers = nn.ModuleList(
            [
                relu_cls(),
                maybe_add_spectral_norm(
                    nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                    add_spectral_norm=add_spectral_norm,
                ),
                relu_cls(),
                maybe_add_spectral_norm(
                    nn.Conv2d(n_filters, n_filters, kernel_size, padding=1),
                    add_spectral_norm=add_spectral_norm,
                ),
            ]
        )

    def forward(self, x):
        _x = x
        for op in self.layers:
            _x = op(_x)
        return x + _x


class Generator(nn.Module):
    def __init__(self, n_filters=256):
        super(Generator, self).__init__()
        self.fc = nn.Linear(128, 4 * 4 * 256)
        network = [
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*network)
        self.noise = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, z):
        z = self.fc(z).reshape(-1, 256, 4, 4)
        return self.net(z)

    def sample(self, n_samples):
        z = self.noise.sample([n_samples, 128]).to(ptu.device)
        return self.forward(z)


class Solver(object):
    def __init__(self, train_data, n_iterations=50000, batch_size=256, n_filters=128):
        self.n_critic = 5
        self.log_interval = 100
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.train_loader = self.create_loaders(train_data)
        self.n_batches_in_epoch = len(self.train_loader)
        # Number of epochs to train in order to have n_iterations of the
        # generator, meaning that do a gradient update for the generator
        # n_iterations times.
        self.n_epochs = self.n_critic * n_iterations // self.n_batches_in_epoch
        self.n_iterations = n_iterations
        self.curr_itr = 0

    def build(self, part_name):
        self.g = Generator(n_filters=self.n_filters).to(ptu.device)
        self.d = Discriminator().to(ptu.device)
        self.g_optimizer = torch.optim.Adam(
            self.g.parameters(), lr=2e-4, betas=(0, 0.9)
        )
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.g_optimizer,
            lambda iter: (self.n_iterations - iter) / self.n_iterations,
            last_epoch=-1,
        )
        self.d_optimizer = torch.optim.Adam(
            self.d.parameters(), lr=2e-4, betas=(0, 0.9)
        )
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.d_optimizer,
            lambda iter: (self.n_iterations - iter) / self.n_iterations,
            last_epoch=-1,
        )
        self.part_name = part_name

    def create_loaders(self, train_data):
        train_loader = data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )
        return train_loader

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]

        # Calculate interpolation
        eps = torch.rand(batch_size, 1, 1, 1).to(ptu.device)
        eps = eps.expand_as(real_data)
        interpolated = eps * real_data.data + (1 - eps) * fake_data.data
        interpolated.requires_grad = True

        d_output = self.d(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_output,
            inputs=interpolated,
            grad_outputs=torch.ones(d_output.size()).to(ptu.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def train(self, verbose=False, checkpoint_freq=100):
        train_g_losses = []
        train_d_losses = []
        for epoch_i in tqdm_notebook(range(self.n_epochs), desc="Epoch", leave=False):
            # epoch_i += 1

            self.d.train()
            self.g.train()
            g_batch_loss_history = []
            d_batch_loss_history = []

            for batch_i, x in enumerate(
                tqdm_notebook(self.train_loader, desc="Batch", leave=False)
            ):
                self.curr_itr += 1
                x = ptu.tensor(x).float()
                # Normalize x to [-1, 1].
                x = 2 * (x - 0.5)

                # do a critic update
                self.d_optimizer.zero_grad()
                fake_data = self.g.sample(x.shape[0])
                gp = self.gradient_penalty(x, fake_data)
                d_fake_loss = self.d(fake_data).mean()
                d_real_loss = self.d(x).mean()
                d_loss = d_fake_loss - d_real_loss + 10 * gp
                d_loss.backward()
                self.d_optimizer.step()
                d_batch_loss_history.append(d_loss.data.cpu().numpy())

                # generator update
                if self.curr_itr % self.n_critic == 0:
                    self.g_optimizer.zero_grad()
                    fake_data = self.g.sample(self.batch_size)
                    g_loss = -self.d(fake_data).mean()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # step the learning rate
                    self.g_scheduler.step()
                    self.d_scheduler.step()

                    g_batch_loss_history.append(g_loss.data.cpu().numpy())

                    if verbose:
                        print(
                            f"Epoch {epoch_i}, Batch {batch_i}: D loss: {d_loss.data.cpu().numpy()}, D fake loss: {d_fake_loss.data.cpu().numpy()}, D real loss: {d_real_loss.data.cpu().numpy()}, GP: {gp.data.cpu().numpy()} G loss: {g_loss.data.cpu().numpy()}, G LR: {self.g_scheduler.get_last_lr()}, D LR: {self.d_scheduler.get_last_lr()}"
                        )

            g_epoch_loss = np.mean(g_batch_loss_history)
            train_g_losses.append(g_epoch_loss)
            d_epoch_loss = np.mean(d_batch_loss_history)
            train_d_losses.append(d_epoch_loss)
            np.save("q2_train_g_losses.npy", np.array(train_g_losses))
            np.save("q2_train_d_losses.npy", np.array(train_d_losses))

            # Save a checkpoint.
            if epoch_i % checkpoint_freq == 0:
                self.save_model(f"{self.part_name}_epoch_{epoch_i}.pt")

        train_g_losses = np.array(train_g_losses)
        train_d_losses = np.array(train_d_losses)
        self.save_model(f"{self.part_name}_final.pt")
        return train_g_losses, train_d_losses

    def save_model(self, filename):
        torch.save(self.g.state_dict(), "g_" + filename)
        torch.save(self.d.state_dict(), "d_" + filename)

    def load_model(self, filename):
        d_path = "d_" + filename
        g_path = "g_" + filename
        self.d.load_state_dict(torch.load(d_path))
        self.g.load_state_dict(torch.load(g_path))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        network = [
            ResnetBlockDown(3, n_filters=128),
            ResnetBlockDown(128, n_filters=128),
            ResBlock(128, n_filters=128),
            ResBlock(128, n_filters=128),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*network)
        self.fc = nn.Linear(128, 1)

    def forward(self, z):
        z = self.net(z)
        z = torch.sum(z, dim=(2, 3))
        return self.fc(z)


class VQDiscriminator(nn.Module):
    def __init__(
        self, n_filters=128, leaky_relu=False, patchify=True, add_spectral_norm=False
    ):
        super(VQDiscriminator, self).__init__()
        network = [
            ResnetBlockDown(
                3,
                n_filters=n_filters,
                add_spectral_norm=add_spectral_norm,
                leaky_relu=leaky_relu,
            ),
            ResnetBlockDown(
                n_filters,
                n_filters=n_filters,
                add_spectral_norm=add_spectral_norm,
                leaky_relu=leaky_relu,
            ),
            ResBlock(
                n_filters,
                n_filters=n_filters,
                add_spectral_norm=add_spectral_norm,
                leaky_relu=leaky_relu,
            ),
            ResBlock(
                n_filters,
                n_filters=n_filters,
                add_spectral_norm=add_spectral_norm,
                leaky_relu=leaky_relu,
            ),
        ]
        if leaky_relu:
            network.append(nn.LeakyReLU())
        else:
            network.append(nn.ReLU())
        self.net = nn.Sequential(*network)
        self.fc = maybe_add_spectral_norm(
            nn.Linear(n_filters, 1), add_spectral_norm=add_spectral_norm
        )

        # self.process_z = self.patchify if patchify else nn.Identity()

    def patchify(self, z):
        bs, nc, h, w = z.shape
        # Split z into 8x8 patches
        z = z.unfold(2, 8, 8).unfold(3, 8, 8)
        z = (
            z.contiguous()
            .view(bs, nc, -1, 8, 8)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, nc, 8, 8)
        )
        return z

    def forward(self, z):
        if self.patchify:
            z = self.patchify(z)

        z = self.net(z)
        z = torch.sum(z, dim=(2, 3))
        return torch.sigmoid(self.fc(z))


class VQGAN(object):
    def __init__(
        self,
        train_data,
        val_data,
        reconstruct_data,
        batch_size=256,
        n_filters=128,
        code_dim=256,
        code_size=1024,
        num_epochs=15,
        vit_vqgan: bool = False,
        use_scheduler: bool = True,
    ):
        self.train_loader = self.create_loaders(train_data, batch_size, shuffle=True)
        self.val_loader = self.create_loaders(val_data, batch_size, shuffle=False)
        self.reconstruct_loader = self.create_loaders(
            reconstruct_data, batch_size, shuffle=False
        )

        # Initialize models, optimizers, and loss functions
        self.vit_vqgan = vit_vqgan
        if vit_vqgan:
            quantizer_kwargs = {"beta": 1, "use_norm": True, "embed_init": "uniform"}
            self.generator = vit.ViTVQ(
                image_size=32, patch_size=4, quantizer_kwargs=quantizer_kwargs
            ).to(ptu.device)
            self.discriminator = VQDiscriminator(
                n_filters=n_filters,
                leaky_relu=True,
                patchify=False,
                add_spectral_norm=True,
            ).to(ptu.device)
            g_lr = 1e-3  # 2e-4
            betas = (0.9, 0.99)  # (0.5, 0.9)
        else:
            self.generator = vqvae.VQVAE(code_dim=code_dim, code_size=code_size).to(
                ptu.device
            )
            self.discriminator = VQDiscriminator(
                n_filters=n_filters,
                leaky_relu=False,
                patchify=True,
                add_spectral_norm=False,
            ).to(ptu.device)
            g_lr = 2e-4
            betas = (0.5, 0.9)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr, betas=betas)
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=betas
        )

        self.num_epochs = num_epochs
        self.use_scheduler = use_scheduler
        num_batches_in_epoch = len(self.train_loader)
        self.n_iterations = num_epochs * num_batches_in_epoch
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.g_optimizer,
            lambda iter: (self.n_iterations - iter) / self.n_iterations,
            last_epoch=-1,
        )
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.d_optimizer,
            lambda iter: (self.n_iterations - iter) / self.n_iterations,
            last_epoch=-1,
        )

        self.lpips_loss_fn = LPIPS().eval().to(ptu.device)

    def create_loaders(self, data, batch_size, shuffle=True):
        data = (data * 2) - 1
        dataset = TensorDataset(torch.tensor(data).float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(self, verbose=False, log_freq=100):
        # Placeholder for losses
        discriminator_losses = []
        l_pips_losses = []
        l2_recon_train = []
        l2_recon_test = []

        iter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            self.generator.train()
            self.discriminator.train()

            for i, (real_data,) in enumerate(
                tqdm_notebook(self.train_loader, desc="Batch", leave=False)
            ):
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                # Training discriminator
                real_data = ptu.tensor(real_data)
                # if self.vit_vqgan:
                #     fake_data, _ = self.generator(real_data)
                # else:
                #     fake_data = self.generator.reconstruct(real_data)
                fake_data, g_reg_loss = self.generator(real_data)
                d_real_loss = -(self.discriminator(real_data) + 1e-8).log().mean()
                d_fake_pred = self.discriminator(fake_data)
                d_fake_loss = -(1 - d_fake_pred + 1e-8).log().mean()
                # d_loss is same as loss_d (ignoring weighting factor)
                d_loss = 0.1 * (d_fake_loss + d_real_loss)
                d_loss.backward(retain_graph=True)
                d_loss_val = d_loss.item()
                discriminator_losses.append(d_loss_val)

                # Training generator
                # fake_data, g_reg_loss = self.generator(real_data)
                # g_gan_loss is same as loss_g
                g_gan_loss = -(d_fake_pred + 1e-8).log().mean()
                l2_loss = nn.MSELoss()(real_data, fake_data)
                lpips_loss = self.lpips_loss_fn(real_data, fake_data).mean()
                g_loss = 0.1 * g_gan_loss + g_reg_loss + 0.5 * lpips_loss + l2_loss
                if self.vit_vqgan:
                    l1_loss = nn.L1Loss()(real_data, fake_data).mean()
                    g_loss += 0.1 * l1_loss
                g_loss.backward()
                l_pips_losses.append(lpips_loss.item())
                l2_recon_train.append(l2_loss.item())

                # Step the optimizers
                self.d_optimizer.step()
                self.g_optimizer.step()

                # Step the learning rate
                if self.use_scheduler:
                    self.g_scheduler.step()
                    self.d_scheduler.step()

                if verbose and iter % log_freq == 0:
                    print(f"Epoch {epoch}, Batch {i}:")
                    print(
                        f"G loss: {g_loss.item()}, "
                        f"G GAN loss: {g_gan_loss.item()},"
                        f"G reg loss: {g_reg_loss.item()}, "
                        f"LPIPS loss: {lpips_loss.item()}, "
                        f"L2 loss: {l2_loss.item()}, "
                        f"L1 loss: {l1_loss.item() if self.vit_vqgan else 'None'}, "
                        f"G LR: {self.g_scheduler.get_last_lr()}"
                    )
                    print(
                        f"D loss: {d_loss_val}, "
                        f"D real loss: {d_real_loss.item()}, "
                        f"D fake loss: {d_fake_loss.item()}, "
                        f"D LR: {self.d_scheduler.get_last_lr()}"
                    )

                iter += 1

            # Evaluate on validation set
            self.generator.eval()
            l2_losses_val = []
            with torch.no_grad():
                for i, (val_data,) in enumerate(self.val_loader):
                    # [batch_size, channels, height, width]
                    val_data = ptu.tensor(val_data)
                    if self.vit_vqgan:
                        # [batch_size, channels, height, width]
                        recon_val_data, _ = self.generator(val_data)
                    else:
                        # [batch_size, height, width, channels]
                        recon_val_data = self.generator.decode(
                            self.generator.encode(val_data)
                        )
                        recon_val_data = ptu.tensor(recon_val_data).permute(0, 3, 1, 2)
                    l2_loss_val = nn.MSELoss()(val_data, recon_val_data)
                    l2_losses_val.append(l2_loss_val.item())
            l2_recon_test.append(np.mean(l2_losses_val))

        return discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test

    def reconstruct(self):
        reconstructions = []
        self.generator.eval()
        for i, (data,) in enumerate(self.reconstruct_loader):
            data = ptu.tensor(data)
            if self.vit_vqgan:
                with torch.no_grad():
                    # [batch_size, channels, height, width]
                    recon_data, _ = self.generator(data)
                    # [batch_size, height, width, channels]
                    recon_data = recon_data.permute(0, 2, 3, 1).cpu().numpy()
            else:
                # [batch_size, height, width, channels]
                recon_data = self.generator.decode(self.generator.encode(data))
            reconstructions.append(recon_data)

        reconstructions = np.concatenate(reconstructions, axis=0)
        assert reconstructions.min() >= -1 and reconstructions.max() <= 1
        reconstructions = np.clip(reconstructions, -1, 1)
        # Normalize to [0, 1].
        reconstructions = (reconstructions + 1) / 2
        return reconstructions


class MNISTDataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]

        if self.transform:
            image = self.transform(image)

        return image


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        # TODO: this is equivalent to instance normalization for batch size 1,
        # but convert it to nn.InstanceNorm2d just in case.
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        # TODO: this is equivalent to instance normalization for batch size 1,
        # but convert it to nn.InstanceNorm2d just in case.
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):
    """Generator for transfering from cmnist to mnist"""

    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x):
        # print("G12 x.shape:", x.shape)  # (?, 3, 28, 28)
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 14, 14)
        # print("G12 out.shape:", out.shape)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 7, 7)
        # print("G12 out.shape:", out.shape)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        # print("G12 out.shape:", out.shape)
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )
        # print("G12 out.shape:", out.shape)

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 14, 14)
        # print("G12 out.shape:", out.shape)
        out = F.tanh(self.deconv2(out))  # (?, 3, 28, 28)
        # print("G12 out.shape:", out.shape)
        return out


class G21(nn.Module):
    """Generator for transfering from mnist to cmnist"""

    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        # print("G21 x.shape:", x.shape)  # (?, 1, 28, 28)
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 14, 14)
        # print("G21 out.shape:", out.shape)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 7, 7)
        # print("G21 out.shape:", out.shape)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        # print("G21 out.shape:", out.shape)
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )
        # print("G21 out.shape:", out.shape)

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 14, 14)
        # print("G21 out.shape:", out.shape)
        out = F.tanh(self.deconv2(out))  # (?, 3, 28, 28)
        # print("G21 out.shape:", out.shape)
        return out


class D1(nn.Module):
    """Discriminator for cmnist."""

    def __init__(self, conv_dim=64, k_size=3, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(3, conv_dim, k_size, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, k_size)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, k_size)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, k_size, 1, 0, False)

    def forward(self, x):
        # print(x.shape)  # (?, 3, 28, 28)
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 14, 14)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 7, 7)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return torch.sigmoid(out)


class D2(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, k_size=3, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(1, conv_dim, k_size, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, k_size)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, k_size)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, k_size, 1, 0, False)

    def forward(self, x):
        # print(x.shape)  # (?, 1, 28, 28)
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 14, 14)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 7, 7)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return torch.sigmoid(out)  # (2, 2)


def normalize_img(img):
    return (img * 2) - 1


def unnormalize_img(img):
    return (img + 1) / 2


class CycleGANSolver(object):
    def __init__(
        self,
        mnist_data,
        cmnist_data,
        g_conv_dim: int = 64,
        d_conv_dim: int = 64,
        train_iters: int = 40_000,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        log_step: int = 10,
        num_samples: int = 20,
        model_path: str = "cyclegan_models/",
    ):
        self.beta1 = beta1
        self.beta2 = beta2

        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.train_iters = train_iters
        self.lr = lr
        self.log_step = log_step
        self.model_path = model_path

        self.mnist_loader, self.cmnist_loader = self.get_loaders(
            mnist_data, cmnist_data
        )
        self.sample_mnist_loader, self.sample_cmnist_loader = self.get_loaders(
            mnist_data[:num_samples], cmnist_data[:num_samples]
        )
        self.build_model()

    def get_loaders(self, mnist_data, cmnist_data):
        """Creates the data loaders for the mnist and cmnist datasets."""
        # mnist has 1 channel and cmnist has 3 channels
        # mnist_transform = transforms.Compose(
        #     [
        #         transforms.Normalize((0.5,), (0.5,)),
        #     ]
        # )
        # cmnist_transform = transforms.Compose(
        #     [
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )

        # mnist_dataset = MNISTDataset(
        #     torch.tensor(mnist_data).float(), transform=mnist_transform
        # )
        # cmnist_dataset = MNISTDataset(
        #     torch.tensor(cmnist_data).float(), transform=cmnist_transform
        # )

        assert mnist_data.min() >= 0 and mnist_data.max() <= 1
        # Normalize to [-1, 1].
        mnist_data = normalize_img(mnist_data)
        assert mnist_data.min() >= -1 and mnist_data.max() <= 1

        assert cmnist_data.min() >= 0 and cmnist_data.max() <= 1
        # Normalize to [-1, 1].
        cmnist_data = normalize_img(cmnist_data)
        assert cmnist_data.min() >= -1 and cmnist_data.max() <= 1

        mnist_dataset = data.TensorDataset(torch.tensor(mnist_data).float())
        cmnist_dataset = data.TensorDataset(torch.tensor(cmnist_data).float())

        mnist_loader = data.DataLoader(
            dataset=mnist_dataset,
            batch_size=1,
            shuffle=True,
        )
        cmnist_loader = data.DataLoader(
            dataset=cmnist_dataset,
            batch_size=1,
            shuffle=True,
        )

        return mnist_loader, cmnist_loader

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12 = G12(conv_dim=self.g_conv_dim).to(ptu.device)
        self.g21 = G21(conv_dim=self.g_conv_dim).to(ptu.device)
        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=False).to(ptu.device)
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=False).to(ptu.device)

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])

    # TODO: delete?
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h : (i + 1) * h, (j * 2) * h : (j * 2 + 1) * h] = s
            merged[:, i * h : (i + 1) * h, (j * 2 + 1) * h : (j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        """Converts numpy to variable."""
        return ptu.tensor(x, requires_grad=True)

    def to_data(self, x):
        """Converts variable to numpy."""
        return x.cpu().data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        mnist_iter = iter(self.mnist_loader)
        cmnist_iter = iter(self.cmnist_loader)
        iter_per_epoch = min(len(mnist_iter), len(cmnist_iter))

        # fixed mnist and cmnist for sampling
        fixed_mnist = self.to_var(next(mnist_iter)[0])
        fixed_cmnist = self.to_var(next(cmnist_iter)[0])

        for step in range(self.train_iters + 1):
            # reset data_iter for each epoch
            if (step + 1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                cmnist_iter = iter(self.cmnist_loader)

            # load svhn and mnist dataset
            mnist = ptu.tensor(next(mnist_iter)[0])
            cmnist = ptu.tensor(next(cmnist_iter)[0])
            # svhn, s_labels = mnist_iter.next()
            # svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
            # mnist, m_labels = cmnist_iter.next()
            # mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            # ============ train D ============#

            # train with real images
            self.reset_grad()
            # print("cmnist.shape:", cmnist.shape)
            d1_cmnist_pred = self.d1(cmnist)
            # print("d1_cmnist_pred.shape:", d1_cmnist_pred.shape)
            d1_loss = torch.mean((d1_cmnist_pred - 1) ** 2)

            d2_mnist_pred = self.d2(mnist)
            d2_loss = torch.mean((d2_mnist_pred - 1) ** 2)

            d_cmnist_loss = d1_loss
            d_mnist_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_mnist = self.g12(cmnist)
            d2_fake_mnist_pred = self.d2(fake_mnist)
            d2_loss = torch.mean(d2_fake_mnist_pred**2)

            fake_cmnist = self.g21(mnist)
            d1_fake_cmnist_pred = self.d1(fake_cmnist)
            d1_loss = torch.mean(d1_fake_cmnist_pred**2)

            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            # ============ train G ============#

            # train cmnist-mnist-cmnist cycle
            self.reset_grad()
            fake_mnist = self.g12(cmnist)
            d2_fake_mnist_pred = self.d2(fake_mnist)
            recon_cmnist = self.g21(fake_mnist)
            # GAN loss
            backward_gan_loss = torch.mean((d2_fake_mnist_pred - 1) ** 2)
            # Reconstruction loss
            backward_recon_loss = torch.mean((cmnist - recon_cmnist) ** 2)
            backward_g_loss = backward_gan_loss + backward_recon_loss
            backward_g_loss.backward()
            self.g_optimizer.step()

            # train mnist-cmnist-mnist cycle
            self.reset_grad()
            fake_cmnist = self.g21(mnist)
            d1_fake_cmnist_pred = self.d1(fake_cmnist)
            recon_mnist = self.g12(fake_cmnist)
            forward_gan_loss = torch.mean((d1_fake_cmnist_pred - 1) ** 2)
            forward_recon_loss = torch.mean((mnist - recon_mnist) ** 2)
            forward_g_loss = forward_gan_loss + forward_recon_loss
            forward_g_loss.backward()
            self.g_optimizer.step()

            # print the log info
            if (step + 1) % self.log_step == 0:
                print("Step [%d/%d]" % (step + 1, self.train_iters))
                print(
                    "forward_g_loss: %.4f, forward_gan_loss: %.4f, forward_recon_loss: %.4f"
                    % (
                        forward_g_loss.item(),
                        forward_gan_loss.item(),
                        forward_recon_loss.item(),
                    )
                )
                print(
                    "backward_g_loss: %.4f, backward_gan_loss: %.4f, backward_recon_loss: %.4f"
                    % (
                        backward_g_loss.item(),
                        backward_gan_loss.item(),
                        backward_recon_loss.item(),
                    )
                )
                print(
                    "d_real_loss: %.4f, d_cmnist_loss: %.4f, d_mnist_loss: %.4f, "
                    "d_fake_loss: %.4f"
                    % (
                        d_real_loss.item(),
                        d_cmnist_loss.item(),
                        d_mnist_loss.item(),
                        d_fake_loss.item(),
                    )
                )
                print()

    def sample(self):
        mnist_samples = []
        fake_cmnist_samples = []
        mnist_recon_samples = []
        for (mnist,) in self.sample_mnist_loader:
            mnist = mnist.to(ptu.device)
            fake_cmnist = self.g21(mnist)
            mnist_recon = self.g12(fake_cmnist)
            mnist_samples.append(unnormalize_img(mnist.detach().cpu().numpy()))
            fake_cmnist_samples.append(
                unnormalize_img(fake_cmnist.detach().cpu().numpy())
            )
            mnist_recon_samples.append(
                unnormalize_img(mnist_recon.detach().cpu().numpy())
            )

        cmnist_samples = []
        fake_mnist_samples = []
        cmnist_recon_samples = []
        for (cmnist,) in self.sample_cmnist_loader:
            cmnist = cmnist.to(ptu.device)
            fake_mnist = self.g12(cmnist)
            cmnist_recon = self.g21(fake_mnist)
            cmnist_samples.append(unnormalize_img(cmnist.detach().cpu().numpy()))
            fake_mnist_samples.append(
                unnormalize_img(fake_mnist.detach().cpu().numpy())
            )
            cmnist_recon_samples.append(
                unnormalize_img(cmnist_recon.detach().cpu().numpy())
            )

        mnist_samples = np.concatenate(mnist_samples, axis=0).transpose(0, 2, 3, 1)
        fake_cmnist_samples = np.concatenate(fake_cmnist_samples, axis=0).transpose(
            0, 2, 3, 1
        )
        mnist_recon_samples = np.concatenate(mnist_recon_samples, axis=0).transpose(
            0, 2, 3, 1
        )
        cmnist_samples = np.concatenate(cmnist_samples, axis=0).transpose(0, 2, 3, 1)
        fake_mnist_samples = np.concatenate(fake_mnist_samples, axis=0).transpose(
            0, 2, 3, 1
        )
        cmnist_recon_samples = np.concatenate(cmnist_recon_samples, axis=0).transpose(
            0, 2, 3, 1
        )

        return (
            mnist_samples,
            fake_cmnist_samples,
            mnist_recon_samples,
            cmnist_samples,
            fake_mnist_samples,
            cmnist_recon_samples,
        )


def q2(train_data, load=False):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1].
        The first 100 will be displayed, and the rest will be used to calculate the Inception score.
    """
    solver = Solver(train_data, n_iterations=50000)
    solver.build("q2")
    if load:
        solver.load_model("q2_final.pt")
        train_d_losses = np.load("q2_train_d_losses.npy")
    else:
        train_g_losses, train_d_losses = solver.train(verbose=True)

    solver.g.eval()
    solver.d.eval()
    with torch.no_grad():
        samples = solver.g.sample(1000)
        samples = ptu.get_numpy(samples.permute(0, 2, 3, 1)) * 0.5 + 0.5

    return train_d_losses, samples


def q3b(train_data, val_data, reconstruct_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    val_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    reconstruct_data: An (100, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]. To be used for reconstruction

    Returns
    - a (# of training iterations,) numpy array of the discriminator train losses evaluated every minibatch
    - None or a (# of training iterations,) numpy array of the perceptual train losses evaluated every minibatch
    - a (# of training iterations,) numpy array of the l2 reconstruction evaluated every minibatch
    - a (# of epochs + 1,) numpy array of l2 reconstruction loss evaluated once at initialization and after each epoch on the val_data
    - a (100, 32, 32, 3) numpy array of reconstructions from your model in [0, 1] on the reconstruct_data.
    """
    vqgan = VQGAN(
        train_data,
        val_data,
        reconstruct_data,
        num_epochs=30,
        vit_vqgan=True,
        use_scheduler=False,
    )
    discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test = vqgan.train(
        verbose=True, log_freq=50
    )
    reconstructions = vqgan.reconstruct()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Finished at {timestamp}")

    return (
        discriminator_losses,
        l_pips_losses,
        l2_recon_train,
        l2_recon_test,
        reconstructions,
    )


def q4(mnist_data, cmnist_data):
    """
    mnist_data: An (60000, 1, 28, 28) numpy array of black and white images with values in [0, 1]
    cmnist_data: An (60000, 3, 28, 28) numpy array of colored images with values in [0, 1]

    Returns
    - a (20, 28, 28, 1) numpy array of real MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of translated Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of reconstructed MNIST digits, in [0, 1]

    - a (20, 28, 28, 3) numpy array of real Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of translated MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of reconstructed Colored MNIST digits, in [0, 1]
    """
    num_epochs = 5
    solver = CycleGANSolver(
        mnist_data, cmnist_data, train_iters=num_epochs * len(mnist_data), log_step=1000
    )
    solver.train()
    return solver.sample()


def main(args):
    question = args.question
    part = args.part

    if question == 2:
        q2_save_results(q2)
    elif question == 3 and part == "b":
        q3_save_results(q3b, "b")
    elif question == 4:
        q4_save_results(q4)
    else:
        raise NotImplementedError(f"Question {question} is not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 3")
    parser.add_argument("question", type=int, choices=[1, 2, 3, 4], help="Question number")
    parser.add_argument(
        "-p", "--part", choices=["a", "b", "c"], help="Part of the question"
    )
    args = parser.parse_args()
    main(args)
