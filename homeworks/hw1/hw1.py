import argparse
import itertools
import os
import time
from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import deepul.utils as deepul_utils
from deepul import hw1_helper
from deepul.hw1_helper import (  # Q1; Q2; Q3; Q4; Q5; Q6
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    q2a_save_results,
    q2b_save_results,
    q3ab_save_results,
    q3c_save_results,
    q4a_save_results,
    q4b_save_results,
    q5a_save_results,
    q6a_save_results,
    visualize_q1_data,
    visualize_q2a_data,
    visualize_q2b_data,
    visualize_q5_data,
    visualize_q6_data,
)


def eval(model, dataloader, device="cuda"):
    model = model.to(device)
    model.eval()

    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        loss = model.loss(batch)
        total_loss += loss.item() * batch.shape[0]

    return total_loss / len(dataloader.dataset)


def train(
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    learning_rate,
    device="cuda",
    verbose=False,
):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    test_losses.append(eval(model, test_dataloader, device))

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_losses.extend(epoch_losses)
        test_losses.append(eval(model, test_dataloader, device))

        if (epoch + 1) % 10 == 0 and verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    return train_losses, test_losses


def sweep(model, get_data_fn, epochs_list, lrs, batch_sizes, verbose=False):
    hparams_to_loss = {}

    train_data, test_data = get_data_fn()

    for lr, batch_size, epochs in itertools.product(lrs, batch_sizes, epochs_list):
        print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")

        train_dataloader = data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_dataloader = data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

        train_losses, test_losses = train(
            model, train_dataloader, test_dataloader, epochs, lr, verbose=verbose
        )
        last_train_loss = train_losses[-1]
        last_test_loss = test_losses[-1]
        hparams_to_loss[(lr, batch_size, epochs)] = (last_train_loss, last_test_loss)

    return hparams_to_loss


class Histogram(nn.Module):
    def __init__(self, d=30):
        super().__init__()
        self.d = d
        self.logits = nn.Parameter(torch.zeros(d))

    def loss(self, x):
        # Shape: [batch_size, d]
        logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1)
        return F.cross_entropy(logits, x.long())

    def probabilities(self):
        return F.softmax(self.logits, dim=-1)


class MixtureOfLogistics(nn.Module):
    def __init__(self, d, num_mix=4):
        super().__init__()
        self.d = d
        self.num_mix = num_mix

        # pi_i
        self.logits = nn.Parameter(torch.zeros(num_mix))
        # mu_i
        self.means = nn.Parameter(
            torch.arange(num_mix, dtype=torch.float32) / (num_mix - 1) * d
        )
        # Log of scale s_i
        # self.log_scales = nn.Parameter(torch.randn(num_mix))
        self.log_scales = nn.Parameter(torch.zeros(num_mix))

        # Tolerance for log probability near 0 and d - 1.
        self._log_prob_tol = 1e-3
        # Min value for CDF for numerical stability when taking the log.
        self._cdf_min = 1e-12

    def forward(self, x):
        x = x.float()
        # [batch_size, num_mix]
        x = x.unsqueeze(1).repeat(1, self.num_mix)
        # [1, num_mix]
        means, log_scales = self.means.unsqueeze(0), self.log_scales.unsqueeze(0)
        log_scales = torch.clamp(log_scales, min=-10.0)
        # 1 / s_i
        inv_scales = torch.exp(-log_scales)

        # CDF of logistics at x + 0.5
        sigmoid_top = torch.sigmoid(inv_scales * (x + 0.5 - means))
        # CDF of logistics at x - 0.5
        sigmoid_bottom = torch.sigmoid(inv_scales * (x - 0.5 - means))
        cdf_delta = sigmoid_top - sigmoid_bottom
        # Log CDF
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=self._cdf_min))

        # Log CDF for x = 0
        log_cdf_max = torch.log(
            torch.clamp(torch.sigmoid(inv_scales * (0.5 - means)), min=self._cdf_min)
        )
        # Log CDF for x = d - 1 = 99
        log_cdf_min = torch.log(
            torch.clamp(
                1 - torch.sigmoid(inv_scales * (self.d - 1 - 0.5 - means)),
                min=self._cdf_min,
            )
        )
        # Replace x = d - 1 with log CDF for x = d - 1
        x_log_probs = torch.where(
            x > self.d - 1 - self._log_prob_tol, log_cdf_min, log_cdf_delta
        )
        # Replace x = 0 with log CDF for x = 0
        x_log_probs = torch.where(x < self._log_prob_tol, log_cdf_max, x_log_probs)
        pi_log_probs = F.log_softmax(self.logits, dim=0).unsqueeze(0)
        log_probs = x_log_probs + pi_log_probs
        return torch.logsumexp(log_probs, dim=1)

    def loss(self, x):
        # The forward pass on x returns log probabilities, so we take the mean
        # and negate it to get negative log likelihood loss.
        return -torch.mean(self(x))

    def probabilities(self):
        with torch.no_grad():
            bins = torch.arange(self.d, dtype=torch.float32, device=self.logits.device)
            return torch.exp(self(bins))


MaskType = Literal["A", "B"]


class MaskConv2d(nn.Conv2d):
    def __init__(
        self,
        mask_type: MaskType,
        *args,
        color_conditioning=False,
        **kwargs,
    ):
        """2D Convolution with masked weight for AutoRegressive connection.

        Args:
            mask_type: Either "A" or "B". Determines which weights of the filter
                are used in the convolution.
            *args: Forwarded to `nn.Conv2d`.
            color_conditioning: Whether to use color conditioning or not.
            **kwargs: Forwarded to `nn.Conv2d`.
        """
        if mask_type not in ["A", "B"]:
            raise ValueError(f'Invalid mask type "{mask_type}"')

        super().__init__(*args, **kwargs)

        self.color_conditioning = color_conditioning

        # Make the mask a buffer since it's not a trainable parameter.
        # Shape: [out_channels, in_channels, kernel_size[0], kernel_size[1]
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self._create_mask(mask_type)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.mask,  # Apply the mask to the weights.
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def _create_mask(self, mask_type):
        k = self.kernel_size[0]
        # Only allow (context) connections to pixels above and to the right.
        self.mask[:, :, : k // 2] = 1
        self.mask[:, :, k // 2, : k // 2] = 1

        if self.color_conditioning:
            if self.in_channels % 3 != 0:
                raise ValueError(
                    "Color conditioning can only be used when input has 3 channels."
                )
            if self.out_channels % 3 != 0:
                raise ValueError(
                    "Color conditioning can only be used when output has 3 channels."
                )

            one_third_in = self.in_channels // 3
            one_third_out = self.out_channels // 3
            if mask_type == "B":
                # Allow connection from the red channel to the red channel.
                self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
                # Allow connection from the green channel to the red and green
                # channels.
                self.mask[
                    one_third_out : 2 * one_third_out,
                    : 2 * one_third_in,
                    k // 2,
                    k // 2,
                ] = 1
                # Allow connection from the blue channel to the red, green, and
                # blue channels.
                self.mask[2 * one_third_out :, :, k // 2, k // 2] = 1
            else:
                # Allow connection from the green channel to the red channel.
                self.mask[
                    one_third_out : 2 * one_third_out, :one_third_in, k // 2, k // 2
                ] = 1
                # Alllow connection from the blue channel to the red and green
                # channels.
                self.mask[2 * one_third_out :, : 2 * one_third_in, k // 2, k // 2] = 1
        else:
            if mask_type == "B":
                # Allow connection from a channel to itself.
                self.mask[:, :, k // 2, k // 2] = 1


class ResBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.block = nn.ModuleList(
            [
                nn.ReLU(),
                MaskConv2d("B", in_channels, in_channels // 2, 1, **kwargs),
                nn.ReLU(),
                # Use 7x7 convolution instead of 3x3 convolution (as in the original paper).
                MaskConv2d(
                    "B", in_channels // 2, in_channels // 2, 7, padding=3, **kwargs
                ),
                nn.ReLU(),
                MaskConv2d("B", in_channels // 2, in_channels, 1, **kwargs),
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.block:
            out = layer(out)
        return out + x


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class PixelCNN(nn.Module):
    def __init__(
        self,
        shape,
        num_colors,
        num_filters=64,
        kernel_size=7,
        num_layers=5,
        use_resblock=False,
        color_conditioning=False,
    ):
        """Initialize PixelCNN model.

        Args:
            shape: Shape of the input data in the order [num_channels, height, width].
            num_colors: Number of color categories in each channel. This is the
                number of colors that the dataset is quantized into. For example,
                if num_colors=4, then there are 4 colors and 2 bits per channel.
            num_filters: Number of convolutional filters.
            kernel_size: Kernel size for initial convolutional layers.
            num_layers: Number of masked type B convolutional layers.
            use_resblock: Whether to use ResBlocks instead of MaskedConv2d.
            color_conditioning: Whether to use color conditioning or not.
        """
        super().__init__()

        self.shape = shape
        self.num_colors = num_colors
        self.num_channels = shape[0]
        self.color_conditioning = color_conditioning

        if use_resblock:
            block_init = lambda: ResBlock(
                num_filters, color_conditioning=color_conditioning
            )
        else:
            block_init = lambda: MaskConv2d(
                "B",
                num_filters,
                num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                color_conditioning=color_conditioning,
            )

        model = nn.ModuleList(
            [
                MaskConv2d(
                    "A",
                    self.num_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    color_conditioning=color_conditioning,
                )
            ]
        )
        for _ in range(num_layers):
            if color_conditioning:
                model.append(LayerNorm(color_conditioning, num_filters // 3))
            else:
                model.append(LayerNorm(color_conditioning, num_filters))
            model.extend([nn.ReLU(), block_init()])
        model.extend(
            [
                nn.ReLU(),
                MaskConv2d(
                    "B",
                    num_filters,
                    num_filters,
                    1,
                    color_conditioning=color_conditioning,
                ),
            ]
        )
        model.extend(
            [
                nn.ReLU(),
                MaskConv2d(
                    "B",
                    num_filters,
                    num_colors * self.num_channels,
                    1,
                    color_conditioning=color_conditioning,
                ),
            ]
        )

        self.net = model

    def forward(self, x):
        batch_size = x.shape[0]
        out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5
        for layer in self.net:
            out = layer(out)

        if self.color_conditioning:
            # Shape: [batch_size, num_colors, num_channels, height, width]
            return out.view(
                batch_size, self.num_channels, self.num_colors, *self.shape[1:]
            ).permute(0, 2, 1, 3, 4)
        else:
            # Shape: [batch_size, num_colors, height, width]
            return out.view(batch_size, self.num_colors, *self.shape)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, n):
        """Samples `n` images from the model.
        Args:
            n: Number of images to sample.

        Returns:
            Samples of shape [n, height, width, num_channels].
        """
        # Shape: [n, num_channels, height, width]
        samples = torch.zeros(n, *self.shape).cuda()
        with torch.no_grad():
            for r in range(self.shape[1]):
                for c in range(self.shape[2]):
                    for k in range(self.num_channels):
                        logits = self(samples)[:, :, k, r, c]
                        probs = F.softmax(logits, dim=1)
                        samples[:, k, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.permute(0, 2, 3, 1).cpu().numpy()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear layers that project the input to Q, K, and V for all heads.
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # Create a mask for autoregressive property.
        # 0 means valid position, 1 means masked position.
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, -1e9)
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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Exponentiate at the end for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: [batch_size=1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Shape: [batch_size, seq_len, d_model]
        return self.pe[:, : x.size(1)].repeat(x.size(0), 1, 1)

    def encoding_for_last_token(self, x):
        return self.pe[:, x.size(1) : x.size(1) + 1].repeat(x.size(0), 1, 1)


class TransformerBlock(nn.Module):
    def __init__(self, seq_len, d_model, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=False):
        attn_output = self.attention(x, x, x, use_cache=use_cache)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

    def clear_cache(self):
        self.attention.clear_cache()

    def is_cache_empty(self):
        return self.attention.is_cache_empty()


class Transformer(nn.Module):
    def __init__(self, seq_len, vocab_size, num_layers, d_model, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(seq_len, d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, use_cache=False):
        # x initially has shape [batch_size, seq_len]
        # Only pass the last token through the network if using cache and the
        # cache is not empty for all of the transformer blocks. The cache may be
        # empty if the model is being conditioned on a partial sequence.
        if use_cache and not any(layer.is_cache_empty() for layer in self.layers):
            pos_encoding = self.pos_encoding.encoding_for_last_token(x)
            x = x[:, -1:]
        else:
            pos_encoding = self.pos_encoding(x)

        # Shape: [batch_size, seq_len, d_model]
        x = self.embedding(x) + pos_encoding
        for layer in self.layers:
            x = layer(x, use_cache=use_cache)
        # Shape: [batch_size, seq_len, vocab_size]
        x = self.fc(self.norm(x))
        # Shape: [batch_size, vocab_size, seq_len]
        return x.transpose(1, 2)

    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()


def train_transformer(
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    lr,
    device="cuda",
    verbose=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * epochs
    )

    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for test_batch in test_dataloader:
                test_batch = test_batch.to(device).long()
                logits = model(test_batch)
                test_loss += (
                    criterion(logits[..., :-1], test_batch[..., 1:]).item()
                    * test_batch.shape[0]
                )
            test_loss /= len(test_dataloader.dataset)
            test_losses.append(test_loss)

        model.train()
        total_loss = 0
        for batch_idx, train_batch in enumerate(train_dataloader):
            train_batch = train_batch.to(device).long()
            optimizer.zero_grad()
            logits = model(train_batch)
            loss = criterion(logits[..., :-1], train_batch[..., 1:])
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            train_losses.append(loss_value)
            total_loss += loss_value

        if verbose:
            print(
                f"Epoch {epoch}, Train Loss: {total_loss / len(train_dataloader)}, Test Loss: {test_loss}"
            )

    return train_losses, test_losses


def sample(
    model,
    num_samples,
    seq_len,
    device,
    bos_token,
    use_cache=True,
    prevent_sampling_bos=True,
):
    model.eval()
    if use_cache:
        model.clear_cache()

    # [num_samples, seq_len]
    samples = torch.full(
        (num_samples, 1), fill_value=bos_token, dtype=torch.long, device=device
    )  # Initialize with <bos> token

    time_list = []

    for i in range(seq_len):
        start = time.time()
        # Shape: [num_samples, vocab_size, seq_len]
        logits = model(samples, use_cache=use_cache)
        time_delta = time.time() - start
        if i > 0:
            time_list.append(time_delta)
        if prevent_sampling_bos:
            # Prevent sampling the <bos> token again
            logits[:, bos_token] = -1e10
        # Shape: [num_samples, vocab_size, seq_len]
        probs = F.softmax(logits, dim=1)
        # Shape: [num_samples, seq_len] (seq_len = 1 if use_cache=True)
        next_token = torch.multinomial(probs[:, :, -1], 1)
        samples = torch.cat([samples, next_token], dim=1)

    # Remove the BOS token.
    samples = samples[:, 1:]

    return np.asarray(samples.cpu()), time_list


class ColoredImageTokenizer:
    def __init__(self, num_colors_per_channel=4):
        self.base = num_colors_per_channel

    def encode(self, x):
        # x has shape [batch_size, height, width, num_channels]
        num_channels = x.shape[-1]
        base_arr = np.array([self.base**i for i in range(num_channels)])
        # Shape: [batch_size, height, width]
        return x.dot(base_arr)

    def decode(self, x):
        # x has shape [batch_size, height, width]
        x_decoded = np.zeros(x.shape + (3,), dtype=np.float32)
        x_decoded[..., 0] = x % self.base
        x_decoded[..., 1] = (x // self.base) % self.base
        x_decoded[..., 2] = (x // (self.base**2)) % self.base
        return x_decoded


class CharTokenizedTextDataset(data.Dataset):
    def __init__(self, data, context_length=128, token_to_id=None, id_to_token=None):
        self.bos_id = 0
        self.eos_id = 1
        self.token_to_id = {"<bos>": self.bos_id, "<eos>": self.eos_id}
        self.id_to_token = {self.bos_id: "<bos>", self.eos_id: "<eos>"}
        self.context_length = context_length
        self.encoded_data = []

        if token_to_id is not None and id_to_token is not None:
            self.token_to_id = token_to_id
            self.id_to_token = id_to_token
        else:
            self.build_vocabulary(data)

        self.tokenize_data(data)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def build_vocabulary(self, data):
        unique_chars = set(char for sequence in data for char in sequence)
        for i, char in enumerate(sorted(unique_chars), start=2):
            self.token_to_id[char] = i
            self.id_to_token[i] = char

    def tokenize_data(self, data):
        for sequence in data:
            tokenized_sequence = [self.token_to_id["<bos>"]]
            tokenized_sequence += [self.token_to_id[char] for char in sequence]
            tokenized_sequence += [self.token_to_id["<eos>"]]

            # Split into subsequences of the desired context_length
            for i in range(0, len(tokenized_sequence), self.context_length - 1):
                subsequence = tokenized_sequence[i : i + self.context_length]
                # If the subsequence is too short, pad it with <eos> tokens
                if len(subsequence) < self.context_length:
                    subsequence += [self.token_to_id["<eos>"]] * (
                        self.context_length - len(subsequence)
                    )
                self.encoded_data.append(subsequence)

    def decode(self, sequence, remove_eos=True):
        sequence = list(sequence)
        if remove_eos:
            if self.eos_id in sequence:
                eos_idx = sequence.index(self.eos_id)
                sequence = sequence[:eos_idx]
        tokens = [self.id_to_token[i] for i in sequence]
        return "".join(tokens)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_data[idx], dtype=torch.long)


class MultimodalDataset(data.Dataset):
    def __init__(
        self,
        text_data,
        image_data,
        vqvae,
        quantized_image_shape,
        token_to_id=None,
        id_to_token=None,
        verbose=False,
    ):
        self.bos_token = 0
        self.bos_str = "<bos>"
        self.end_of_text_token = 1
        self.end_of_text_str = "<end of text>"
        self.end_of_image_token = 2
        self.end_of_image_str = "<end of image>"
        self.num_special_tokens = 3

        self.text_data = text_data
        self.image_data = image_data
        self.vqvae = vqvae

        self.quantized_image_shape = quantized_image_shape
        self.image_sequence_length = np.prod(quantized_image_shape)

        self.text_sequence_length = len(self.text_data[0].split())
        self.compute_sequence_length()

        if token_to_id is not None and id_to_token is not None:
            self.token_to_id = token_to_id
            self.id_to_token = id_to_token
        else:
            # Build vocabulary for text
            if verbose:
                print("Building text vocabulary...")
            self.build_text_vocabulary()

        # Adjust image token IDs to not overlap with special and text tokens
        self.image_token_offset = 2 + len(self.token_to_id) + 1

        # Token ranges (start, end (exclusive))
        self.text_token_range = (
            self.num_special_tokens,
            self.num_special_tokens + self.text_vocab_size,
        )
        self.image_token_range = (
            self.text_token_range[1],
            self.text_token_range[1] + self.image_vocab_size,
        )

        # Preprocess data
        if verbose:
            print("Preprocessing data...")
        self.encoded_data = self.preprocess_data()

    @property
    def text_vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def image_vocab_size(self) -> int:
        return self.vqvae.n_embeddings

    @property
    def vocab_size(self) -> int:
        return self.text_vocab_size + self.image_vocab_size + self.num_special_tokens

    def compute_sequence_length(self):
        self.sequence_length = (
            self.num_special_tokens
            + self.text_sequence_length
            + self.image_sequence_length
        )

    def build_text_vocabulary(self):
        word_counts = Counter(
            word for sentence in self.text_data for word in sentence.split()
        )
        self.token_to_id = {}
        for i, word in enumerate(word_counts.keys(), start=self.num_special_tokens):
            self.token_to_id[word] = i
        self.id_to_token = {id: word for word, id in self.token_to_id.items()}

    def encode_image(self, image):
        quantized_image = self.vqvae.quantize(image[np.newaxis, ...]).flatten()
        quantized_image += self.image_token_offset
        return quantized_image

    def condition_on_image(self, image):
        image_tokens = self.encode_image(image)
        return (
            [self.bos_token, self.end_of_text_token]
            + list(image_tokens)
            + [self.end_of_image_token]
        )

    def encode_text(self, text):
        return [self.token_to_id[word] for word in text.split()]

    def condition_on_text(self, text):
        return (
            [self.bos_token, self.end_of_image_token]
            + self.encode_text(text)
            + [self.end_of_text_token]
        )

    def preprocess_data(self):
        encoded_data = []
        for text, image in zip(self.text_data, self.image_data):
            # Tokenize text
            text_tokens = self.encode_text(text)
            # Quantize image and adjust tokens
            image_tokens = self.encode_image(image)

            # Combine text and image sequences
            sequence_ti = (
                [self.bos_token, self.end_of_image_token]
                + text_tokens
                + [self.end_of_text_token]
                + list(image_tokens)
            )
            sequence_it = (
                [self.bos_token, self.end_of_text_token]
                + list(image_tokens)
                + [self.end_of_image_token]
                + text_tokens
            )
            encoded_data.append(sequence_ti)
            encoded_data.append(sequence_it)

        return encoded_data

    def decode(self, samples):
        decoded_samples = []
        for sample in samples:
            if sample[0] == self.end_of_text_token:
                # First modality is image
                quantized_image = sample[1 : 1 + self.image_sequence_length]
                text = sample[2 + self.image_sequence_length :]
            elif sample[0] == self.end_of_image_token:
                # First modality is text
                text = sample[1 : 1 + self.text_sequence_length]
                quantized_image = sample[2 + self.text_sequence_length :]
            else:
                raise ValueError(f"Invalid first token: {samples[0]}")
            quantized_image = quantized_image.reshape(self.quantized_image_shape)[
                np.newaxis, ...
            ]
            quantized_image = quantized_image - self.image_token_offset
            # Decode the image and remove the batch dimension
            image = self.vqvae.decode(quantized_image)[0]
            text = " ".join(self.id_to_token[t] for t in text)
            decoded_samples.append((image, text))

        return decoded_samples

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        sequence = self.encoded_data[idx]
        return torch.tensor(sequence, dtype=torch.long)


def multimodal_sample(
    model,
    train_dataset: MultimodalDataset,
    num_samples,
    seq_len,
    device,
    bos_token,
    end_of_text_token,
    end_of_image_token,
    text_token_range,
    image_token_range,
    text_sequence_length,
    image_sequence_length,
    image_test_prompt=None,
    text_test_prompt=None,
    use_cache=True,
):
    if image_test_prompt is not None and text_test_prompt is not None:
        raise ValueError(
            "Only one of image_test_prompt and text_test_prompt can be set."
        )

    model.eval()
    if use_cache:
        model.clear_cache()

    if image_test_prompt is not None:
        samples = []
        for image in image_test_prompt:
            samples.append(train_dataset.condition_on_image(image))
        samples = torch.tensor(samples, dtype=torch.long, device=device)
        # Initialize modality for each sample, True for text, False for image.
        current_modality = torch.full(
            (num_samples,), True, dtype=torch.bool, device=device
        )
    elif text_test_prompt is not None:
        samples = []
        for text in text_test_prompt:
            samples.append(train_dataset.condition_on_text(text))
        samples = torch.tensor(samples, dtype=torch.long, device=device)
        # Initialize modality for each sample, True for text, False for image.
        current_modality = torch.full(
            (num_samples,), False, dtype=torch.bool, device=device
        )
    else:
        # Initialize samples with <bos> token
        samples = torch.full(
            (num_samples, 1), fill_value=bos_token, dtype=torch.long, device=device
        )
        # Initialize modality for each sample, True for text, False for image.
        # Set to True arbitrarily for step=0.
        current_modality = torch.full(
            (num_samples,), True, dtype=torch.bool, device=device
        )

    # Tracks the number of tokens sampled for the current modality
    tokens_sampled_per_modality = torch.zeros(
        num_samples, dtype=torch.long, device=device
    )

    time_list = []

    # Step start number. 0 for unconditioned samples, greater than 0 for
    # conditioned samples.
    start_step = samples.shape[1] - 1

    for step in range(start_step, seq_len):
        start = time.time()
        # Get logits for the last token only
        # [num_samples, vocab_size, seq_len]
        logits_seq = model(samples, use_cache=use_cache)
        time_delta = time.time() - start
        if step > 0:
            time_list.append(time_delta)
        # [num_samples, vocab_size]
        logits = logits_seq[:, :, -1]

        # Prevent sampling <bos> token again
        logits[:, bos_token] = -1e10

        # Mask logits based on the current modality for each sample
        mask = torch.full_like(logits, fill_value=-1e10)
        for idx, is_text in enumerate(current_modality):
            if step == 0:  # At the start, only allow <end of image> or <end of text>
                mask[idx, [end_of_text_token, end_of_image_token]] = logits[
                    idx, [end_of_text_token, end_of_image_token]
                ]
            elif is_text:
                if tokens_sampled_per_modality[idx] == text_sequence_length:
                    # Force <end of text> token if text sequence length reached
                    mask[idx, end_of_text_token] = 0
                else:
                    # Otherwise, allow only text tokens and <end of text>
                    mask[idx, text_token_range[0] : text_token_range[1]] = logits[
                        idx, text_token_range[0] : text_token_range[1]
                    ]
                    # mask[idx, end_of_text_token] = logits[idx, end_of_text_token]
            else:  # Allow only image tokens and <end of image>
                if tokens_sampled_per_modality[idx] == image_sequence_length:
                    # Force <end of image> token if image sequence length reached
                    mask[idx, end_of_image_token] = 0
                else:
                    # Otherwise, allow only image tokens and <end of image>
                    mask[idx, image_token_range[0] : image_token_range[1]] = logits[
                        idx, image_token_range[0] : image_token_range[1]
                    ]
                    # mask[idx, end_of_image_token] = logits[idx, end_of_image_token]

        logits = mask

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        samples = torch.cat([samples, next_token], dim=1)

        # Update counters and modality
        for idx, token in enumerate(next_token.squeeze(-1)):
            if current_modality[idx]:  # If currently sampling text
                if token == end_of_text_token:
                    current_modality[idx] = False  # Switch to image
                    tokens_sampled_per_modality[idx] = 0  # Reset counter
                elif token != end_of_image_token:
                    tokens_sampled_per_modality[idx] += 1
            else:  # If currently sampling image
                if token == end_of_image_token:
                    current_modality[idx] = True  # Switch to text
                    tokens_sampled_per_modality[idx] = 0  # Reset counter
                elif token != end_of_text_token:
                    tokens_sampled_per_modality[idx] += 1

    # Remove the BOS token from the start of samples
    samples = samples[:, 1:]

    return np.asarray(samples.cpu()), time_list


def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    if dset_id == 1:
        learning_rate = 0.1
        batch_size = 64
        epochs = 100
    elif dset_id == 2:
        learning_rate = 0.1
        batch_size = 128
        epochs = 20

    verbose = False

    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Histogram(d)

    train_losses, test_losses = train(
        model, train_dataloader, test_dataloader, epochs, learning_rate, verbose=verbose
    )
    distribution = model.probabilities().detach().cpu().numpy()

    return train_losses, test_losses, distribution


def q1_b(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    if dset_id == 1:
        batch_size = 256
        epochs = 40
        lr = 1e-1
    elif dset_id == 2:
        batch_size = 8000
        epochs = 1000
        lr = 1e-1

    model = MixtureOfLogistics(d, num_mix=4).cuda()
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    train_losses, test_losses = train(
        model, train_loader, test_loader, epochs, lr, verbose=True
    )
    distribution = model.probabilities().detach().cpu().numpy()

    return train_losses, test_losses, distribution


def q2_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))

    H, W = image_shape
    model = PixelCNN((1, H, W), 2, num_layers=5).cuda()

    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train(model, train_loader, test_loader, 10, 1e-3)
    samples, _ = model.sample(100)
    return train_losses, test_losses, samples


def q2_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """
    # Reshape to [batch_size, C, H, W] for PixelCNN/Conv2D compatibility.
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))

    H, W, C = image_shape
    num_colors = 4
    epochs = 15
    lr = 1e-3
    num_filters = 120
    num_layers = 8
    batch_size = 128
    model = PixelCNN(
        (C, H, W),
        num_colors,
        num_filters=num_filters,
        num_layers=num_layers,
        use_resblock=True,
    ).cuda()

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    train_losses, test_losses = train(model, train_loader, test_loader, epochs, lr)
    samples, _ = model.sample(100)
    return train_losses, test_losses, samples


def q3_a(train_data, test_data, image_shape, dset_id, generate=True):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 1) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, 1), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1, 2, 3}
    """
    np.random.seed(0)
    torch.manual_seed(0)

    height, width, channel = train_data.shape[1:]
    assert channel == 1
    # Shape: [batch_size, seq_len]
    train_data = train_data.reshape(-1, height * width).astype(np.int32)
    test_data = test_data.reshape(-1, height * width).astype(np.int32)

    # Prepend BOS token to the input
    BOS_TOKEN = 2
    train_fill_tensor = np.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = np.concatenate((train_fill_tensor, train_data), axis=1)
    test_fill_tensor = np.full((test_data.shape[0], 1), BOS_TOKEN)
    test_data = np.concatenate((test_fill_tensor, test_data), axis=1)

    if dset_id == 1:
        batch_size = 16
    else:
        batch_size = 64

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    seq_len = train_data.shape[1]
    vocab_size = 3
    epochs = 15
    lr = 1e-3
    num_layers = 2
    d_model = 128
    num_heads = 4
    dropout = 0.0

    model = Transformer(
        seq_len, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    verbose = True

    train_losses, test_losses = train_transformer(
        model, train_loader, test_loader, epochs, lr, verbose=verbose
    )

    if verbose:
        print("Sampling...")

    if generate:
        samples, _ = sample(
            model, 100, height * width, "cuda", BOS_TOKEN, use_cache=True
        )
        samples = samples.reshape(samples.shape[0], height, width)[..., np.newaxis]
    else:
        samples = None

    return train_losses, test_losses, samples, model


def q3_b(train_data, test_data, image_shape, dset_id, generate=True):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """
    if dset_id == 1:
        batch_size = 16
    elif dset_id == 2:
        batch_size = 64

    epochs = 15
    lr = 1e-3
    num_layers = 2
    d_model = 128
    num_heads = 4
    dropout = 0.0
    verbose = True

    np.random.seed(0)
    torch.manual_seed(0)

    height, width, num_channels = train_data.shape[1:]
    assert num_channels == 3

    # Tokenize the input
    tokenizer = ColoredImageTokenizer(4)
    train_data = tokenizer.encode(train_data)
    test_data = tokenizer.encode(test_data)

    # Shape: [batch_size, seq_len]
    train_data = train_data.reshape(-1, height * width).astype(np.int32)
    test_data = test_data.reshape(-1, height * width).astype(np.int32)

    # Prepend BOS token to the input
    BOS_TOKEN = 64
    train_fill_tensor = np.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = np.concatenate((train_fill_tensor, train_data), axis=1)
    test_fill_tensor = np.full((test_data.shape[0], 1), BOS_TOKEN)
    test_data = np.concatenate((test_fill_tensor, test_data), axis=1)

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    seq_len = train_data.shape[1]
    vocab_size = 65

    model = Transformer(
        seq_len, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    train_losses, test_losses = train_transformer(
        model, train_loader, test_loader, epochs, lr, verbose=verbose
    )

    if verbose:
        print("Sampling...")

    if generate:
        num_samples = 100
        # Set seq_len to the size of an image, excluding the BOS token.
        samples, _ = sample(
            model, num_samples, height * width, "cuda", BOS_TOKEN, use_cache=True
        )
        samples = samples.reshape(samples.shape[0], height, width)
        samples = tokenizer.decode(samples)
    else:
        samples = None

    return train_losses, test_losses, samples, model


def q3_c(model, train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """
    num_samples = 100
    height, width = train_data.shape[1], train_data.shape[2]
    BOS_TOKEN = 64

    base = 4
    tokenizer = ColoredImageTokenizer(base)

    samples_lists = []
    time_lists = []
    for use_cache in [True, False]:
        torch.manual_seed(0)
        np.random.seed(0)

        samples, time_list = sample(
            model, num_samples, height * width, "cuda", BOS_TOKEN, use_cache=use_cache
        )
        samples = samples.reshape(samples.shape[0], height, width)
        samples = tokenizer.decode(samples)
        samples_lists.append(samples)
        time_lists.append(time_list)

    time_list_with_cache, time_list_no_cache = time_lists
    samples_with_cache, samples_no_cache = samples_lists

    return (
        time_list_no_cache,
        time_list_with_cache,
        samples_no_cache,
        samples_with_cache,
    )


def q4_a(images, vqvae):
    """
    images: (B, H, W, C), the images to pass through the encoder and decoder of the vqvae
    vqvae: a vqvae model, trained on the relevant dataset

    Returns
    - a numpy array of size (2, H, W, C) of the decoded image
    """
    encoded_images = vqvae.quantize(images)
    autoencoded_images = vqvae.decode(encoded_images)
    return autoencoded_images


def q4_b(train_data, test_data, image_shape, dset_id, vqvae, generate=True, save=True):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets
    vqvae: a vqvae model, trained on dataset dset_id

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """

    if dset_id == 1:
        batch_size = 64
    elif dset_id == 2:
        batch_size = 64

    epochs = 30
    lr = 1e-3
    num_layers = 4
    d_model = 128
    num_heads = 4
    dropout = 0.0
    verbose = True

    np.random.seed(0)
    torch.manual_seed(0)

    # Set to None to use the full dataset
    max_num_train_examples = None
    if max_num_train_examples is not None:
        train_data = train_data[:max_num_train_examples]

    num_train_examples, height, width, num_channels = train_data.shape
    assert num_channels == 3
    num_test_examples = test_data.shape[0]

    # Tokenize the input
    if verbose:
        print("Quantizing...")
    # Shape: [num_train_examples, 7, 7]
    train_data = vqvae.quantize(train_data)
    # Shape: [num_test_examples, 7, 7]
    test_data = vqvae.quantize(test_data)
    quantized_height, quantized_width = train_data.shape[1:]
    assert train_data.shape[0] == num_train_examples
    assert test_data.shape[0] == num_test_examples

    # Shape: [batch_size, seq_len]
    train_data = train_data.reshape(num_train_examples, -1).to(torch.int32)
    test_data = test_data.reshape(num_test_examples, -1).to(torch.int32)

    vocab_size = vqvae.n_embeddings + 1

    # Prepend BOS token to the input
    BOS_TOKEN = vqvae.n_embeddings
    train_fill_tensor = torch.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = torch.cat((train_fill_tensor, train_data), dim=1)
    test_fill_tensor = torch.full((test_data.shape[0], 1), BOS_TOKEN)
    test_data = torch.cat((test_fill_tensor, test_data), dim=1)

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    seq_len = train_data.shape[1]
    model = Transformer(
        seq_len, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    if verbose:
        print("Training...")
    train_losses, test_losses = train_transformer(
        model, train_loader, test_loader, epochs, lr, verbose=verbose
    )

    if verbose:
        print("Sampling...")

    if generate:
        num_samples = 100
        samples, _ = sample(
            model,
            num_samples,
            quantized_height * quantized_width,
            "cuda",
            BOS_TOKEN,
            use_cache=True,
        )
        samples = samples.reshape(samples.shape[0], quantized_height, quantized_width)
        samples = vqvae.decode(samples)
    else:
        samples = None

    return train_losses, test_losses, samples, model


def q5_a(train_text, test_text, generate=True):
    """
    train_text: list[str] Train text sequences.
    test_text: list[str] Test text sequences.

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a list of 5 (str), 5 generated samples from the model.
    """
    batch_size = 64
    context_length = 128
    epochs = 15
    lr = 1e-3
    num_layers = 4
    d_model = 128
    num_heads = 4
    dropout = 0.0
    verbose = True

    np.random.seed(0)
    torch.manual_seed(0)

    # Set to None to use the full dataset
    max_num_train_examples = None
    if max_num_train_examples is not None:
        train_text = train_text[:max_num_train_examples]

    # Make dataloaders
    train_dataset = CharTokenizedTextDataset(train_text, context_length=context_length)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CharTokenizedTextDataset(
        test_text,
        context_length=context_length,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
    )
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = train_dataset.vocab_size

    model = Transformer(
        context_length, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    if verbose:
        print("Training...")
    train_losses, test_losses = train_transformer(
        model, train_loader, test_loader, epochs, lr, verbose=verbose
    )

    if verbose:
        print("Sampling...")

    if generate:
        num_samples = 5
        samples, _ = sample(
            model,
            num_samples,
            # Subtract 1 to account for the <bos> token
            context_length - 1,
            "cuda",
            train_dataset.bos_id,
            use_cache=True,
        )
        text_samples = [
            train_dataset.decode(sequence, remove_eos=True) for sequence in samples
        ]
    else:
        text_samples = None

    return train_losses, test_losses, text_samples, model


def q6_a(
    train_data,
    test_data,
    image_shape,
    train_text,
    test_text,
    image_test_prompt,
    text_test_prompt,
    vqvae,
    generate=True,
):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: tuple (H, W, C) The shape of the images in the dataset, indicating height, width, and number of color channels.
    train_text: list[str] Text data associated with each training image.
    test_text: list[str] Text data associated with each test image.
    image_test_prompt: (9, H, W, C) Image data used for generating conditional text samples during testing.
    text_test_prompt: list of 9 strings Text prompts used for generating conditional image samples during testing.
    vqvae: a vqvae model, trained on the relevant dataset

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a list of 9 (image, text), corresponding to the image conditioned samples
    - a list of 9 (image, text), corresponding to the text conditions samples
    - a list of 9 (image, text), corresponding to unconditional samples
    """
    batch_size = 64
    epochs = 15
    lr = 1e-3
    num_layers = 4
    d_model = 128
    num_heads = 4
    dropout = 0.0
    verbose = True

    np.random.seed(0)
    torch.manual_seed(0)

    # Set to None to use the full dataset
    max_num_examples = None
    if max_num_examples is not None:
        train_data = train_data[:max_num_examples]
        train_text = train_text[:max_num_examples]
        test_data = test_data[:max_num_examples]
        test_text = test_text[:max_num_examples]

    quantized_image_shape = (7, 7)
    train_dataset = MultimodalDataset(
        train_text,
        train_data,
        vqvae,
        quantized_image_shape=quantized_image_shape,
        verbose=True,
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MultimodalDataset(
        test_text,
        test_data,
        vqvae,
        quantized_image_shape=quantized_image_shape,
        token_to_id=train_dataset.token_to_id,
        id_to_token=train_dataset.id_to_token,
        verbose=True,
    )
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = train_dataset.vocab_size

    seq_len = train_dataset.sequence_length
    model = Transformer(
        seq_len, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    if verbose:
        print("Training...")
    train_losses, test_losses = train_transformer(
        model, train_loader, test_loader, epochs, lr, verbose=verbose
    )

    if generate:
        if verbose:
            print("Sampling...")
        num_samples = 9
        samples_unconditioned, _ = multimodal_sample(
            model,
            train_dataset,
            num_samples,
            # Subtract 1 to account for the <bos> token
            seq_len - 1,
            "cuda",
            train_dataset.bos_token,
            train_dataset.end_of_text_token,
            train_dataset.end_of_image_token,
            train_dataset.text_token_range,
            train_dataset.image_token_range,
            train_dataset.text_sequence_length,
            train_dataset.image_sequence_length,
            use_cache=True,
        )
        samples_unconditioned = train_dataset.decode(samples_unconditioned)

        samples_text_conditioned, _ = multimodal_sample(
            model,
            train_dataset,
            num_samples,
            # Subtract 1 to account for the <bos> token
            seq_len - 1,
            "cuda",
            train_dataset.bos_token,
            train_dataset.end_of_text_token,
            train_dataset.end_of_image_token,
            train_dataset.text_token_range,
            train_dataset.image_token_range,
            train_dataset.text_sequence_length,
            train_dataset.image_sequence_length,
            text_test_prompt=text_test_prompt,
            use_cache=True,
        )
        samples_text_conditioned = train_dataset.decode(samples_text_conditioned)

        samples_image_conditioned, _ = multimodal_sample(
            model,
            train_dataset,
            num_samples,
            # Subtract 1 to account for the <bos> token
            seq_len - 1,
            "cuda",
            train_dataset.bos_token,
            train_dataset.end_of_text_token,
            train_dataset.end_of_image_token,
            train_dataset.text_token_range,
            train_dataset.image_token_range,
            train_dataset.text_sequence_length,
            train_dataset.image_sequence_length,
            image_test_prompt=image_test_prompt,
            use_cache=True,
        )
        samples_image_conditioned = train_dataset.decode(samples_image_conditioned)
    else:
        samples_image_conditioned = samples_text_conditioned = samples_unconditioned = (
            None
        )

    return (
        train_losses,
        test_losses,
        samples_image_conditioned,
        samples_text_conditioned,
        samples_unconditioned,
        model,
    )


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Add arguments
    parser.add_argument(
        "-q",
        "--question",
        type=int,
        choices=range(1, 7),
        required=True,
        help="Question number (1 through 6)",
    )
    parser.add_argument(
        "-p",
        "--part",
        type=str,
        choices=["a", "b", "c"],
        required=True,
        help="Part (a, b, or c)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=int,
        choices=[1, 2],
        required=False,
        help="Dataset number (1 or 2)",
    )

    # Parse the arguments
    args = parser.parse_args()
    dataset = args.dataset
    question = args.question
    part = args.part

    print(
        f"Dataset: {args.dataset or 'None'}, Question: {args.question}, Part: {args.part}"
    )

    if question == 1:
        if part == "a" or part == "b":
            print(f"Q 1{part} ds {dataset}")
            fn = q1_a if part == "a" else q1_b
            q1_save_results(dataset, part, fn)
            return
    elif question == 2:
        if part == "a":
            print(f"Q 2a ds {dataset}")
            q2a_save_results(dataset, q2_a)
            return
        elif part == "b":
            print(f"Q 2b ds {dataset}")
            q2b_save_results(dataset, part, q2_b)
            return
    elif question == 3:
        if part == "a" or part == "b":
            print(f"Q 3{part} ds {dataset}")
            fn = q3_a if part == "a" else q3_b
            q3ab_save_results(dataset, part, fn, generate=True, save=True)
            return
        elif part == "c":
            model = q3ab_save_results(dataset, "b", q3_b, generate=False, save=False)[
                -1
            ]
            print(f"Q 3c ds {dataset}")
            q3c_save_results(dataset, q3_c, model)
            return
    elif question == 4:
        if part == "a":
            print(f"Q 4a ds {dataset}")
            q4a_save_results(dataset, q4_a)
            return
        if part == "b":
            print(f"Q 4b ds {dataset}")
            q4b_save_results(dataset, q4_b)
            return
    elif question == 5:
        if part == "a":
            print("Q 5a")
            q5a_save_results(q5_a)
            return
    elif question == 6:
        if part == "a":
            print("Q 6a")
            q6a_save_results(q6_a)
            return

    raise NotImplementedError(f"Question {question} part {part} not implemented")


if __name__ == "__main__":
    main()
