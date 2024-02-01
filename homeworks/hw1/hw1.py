import itertools
import os
import time

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

    # def forward(self, q, k, v, use_cache=False):
    #     """Apply forward pass.

    #     Args:
    #         q: Query tensor of shape (batch_size, seq_len, d_model)
    #         k: Key tensor of shape (batch_size, seq_len, d_model)
    #         v: Value tensor of shape (batch_size, seq_len, d_model)
    #         mask: Mask tensor of shape (batch_size, seq_len).
    #             0 means valid position, 1 means masked position.
    #     """
    #     batch_size = q.size(0)

    #     if use_cache and self.cache is not None:
    #         cache_k = self.cache["k"]
    #         cache_v = self.cache["v"]

    #     q = self.split_heads(self.wq(q), batch_size)
    #     k = self.split_heads(self.wk(k), batch_size)
    #     v = self.split_heads(self.wv(v), batch_size)

    #     scaled_attention_logits = (
    #         torch.matmul(q, k.transpose(-2, -1)) / self.depth**0.5
    #     )

    #     # Apply mask to the scaled attention logits
    #     seq_len = q.size(2)
    #     scaled_attention_logits += self.mask[:seq_len, :seq_len]

    #     attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    #     output = torch.matmul(attention_weights, v)
    #     # Concatenate the output of all heads
    #     output = (
    #         output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
    #     )

    #     return self.dense(output)

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
        scaled_attention_logits = (
            torch.matmul(q, k.transpose(-2, -1)) / self.depth**0.5
        )

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
        if use_cache:
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
    # warmup_steps,
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


def sample(model, num_samples, seq_len, device, bos_token, use_cache=True):
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
        # Shape: [num_samples, vocab_size, seq_len]
        probs = F.softmax(logits, dim=1)
        # Shape: [num_samples, seq_len] (seq_len = 1 if use_cache=True)
        next_token = torch.multinomial(probs[:, :, -1], 1)
        samples = torch.cat([samples, next_token], dim=1)

    # Remove the BOS token.
    samples = samples[:, 1:]

    return np.asarray(samples.cpu()), time_list


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
    fill_tensor = np.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = np.concatenate((fill_tensor, train_data), axis=1)

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
    base = 4
    num_channels = train_data.shape[-1]
    base_arr = np.array([base**i for i in range(num_channels)])
    # Shape: [batch_size, height, width]
    train_data = train_data.dot(base_arr)
    test_data = test_data.dot(base_arr)

    # Shape: [batch_size, seq_len]
    train_data = train_data.reshape(-1, height * width).astype(np.int32)
    test_data = test_data.reshape(-1, height * width).astype(np.int32)

    # Prepend BOS token to the input
    BOS_TOKEN = 64
    fill_tensor = np.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = np.concatenate((fill_tensor, train_data), axis=1)

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
        samples, _ = sample(
            model, num_samples, height * width, "cuda", BOS_TOKEN, use_cache=True
        )
        samples = samples.reshape(samples.shape[0], height, width)
        samples_decoded = np.zeros(samples.shape + (3,), dtype=np.uint8)
        samples_decoded[..., 0] = samples % base
        samples_decoded[..., 1] = (samples // base) % base
        samples_decoded[..., 2] = (samples // (base**2)) % base
        samples = samples_decoded
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

    samples_lists = []
    time_lists = []
    for use_cache in [True, False]:
        torch.manual_seed(0)
        np.random.seed(0)

        samples, time_list = sample(
            model, num_samples, height * width, "cuda", BOS_TOKEN, use_cache=use_cache
        )
        samples = samples.reshape(samples.shape[0], height, width)
        samples_decoded = np.zeros(samples.shape + (3,), dtype=np.uint8)
        samples_decoded[..., 0] = samples % base
        samples_decoded[..., 1] = samples // base
        samples_decoded[..., 2] = samples // (base**2)
        samples_lists.append(samples_decoded)
        time_lists.append(time_list)

    time_list_with_cache, time_list_no_cache = time_lists
    samples_with_cache, samples_no_cache = samples_lists

    return (
        time_list_no_cache,
        time_list_with_cache,
        samples_no_cache,
        samples_with_cache,
    )


if __name__ == "__main__":
    # print("Q 3a ds 1")
    # q3ab_save_results(1, "a", q3_a)
    # print("Q 3a ds 2")
    # q3ab_save_results(2, "a", q3_a)

    print("Q 3b ds 1")
    model_q3b_ds1 = q3ab_save_results(1, "b", q3_b)[-1]
    print("Q 3c ds 1")
    q3c_save_results(1, q3_c, model_q3b_ds1)
    del model_q3b_ds1

    print("Q 3b ds 2")
    model_q3b_ds2 = q3ab_save_results(2, "b", q3_b)[-1]
    print("Q 3c ds 2")
    q3c_save_results(2, q3_c, model_q3b_ds2)
    del model_q3b_ds2
