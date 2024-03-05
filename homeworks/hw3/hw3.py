import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import norm
from tqdm import tqdm, tqdm_notebook, trange

import deepul.pytorch_util as ptu
import deepul.utils as deepul_utils
from deepul.hw3_helper import *

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


class Downsample_Conv2d(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True
    ):
        super(Downsample_Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias
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
    def __init__(self, in_dim, kernel_size=(3, 3), stride=1, n_filters=256):
        super(ResnetBlockDown, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size, stride=stride, padding=1),
                nn.ReLU(),
                Downsample_Conv2d(n_filters, n_filters, kernel_size),
                Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0),
            ]
        )

    def forward(self, x):
        _x = x
        for i in range(len(self.layers) - 1):
            _x = self.layers[i](_x)
        return self.layers[-1](x) + _x


class ResBlock(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size, padding=1),
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


def main(args):
    question = args.question
    part = args.part

    if question == 2:
        q2_save_results(q2)
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
