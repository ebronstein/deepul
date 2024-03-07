import argparse
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim
import torch.utils.data as data
import vit
from einops import repeat
from scipy.stats import norm
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


def main(args):
    question = args.question
    part = args.part

    if question == 2:
        q2_save_results(q2)
    elif question == 3 and part == "b":
        q3_save_results(q3b, "b")
    else:
        raise NotImplementedError(f"Question {question} is not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 3")
    parser.add_argument("question", type=int, choices=[1, 2, 3], help="Question number")
    parser.add_argument(
        "-p", "--part", choices=["a", "b", "c"], help="Part of the question"
    )
    args = parser.parse_args()
    main(args)
