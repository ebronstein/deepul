import argparse
import itertools
import os
import time
from collections import Counter

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
    def __init__(self, data, context_length=128):
        self.bos_id = 0
        self.eos_id = 1
        self.token_to_id = {"<bos>": self.bos_id, "<eos>": self.eos_id}
        self.id_to_token = {self.bos_id: "<bos>", self.eos_id: "<eos>"}
        self.context_length = context_length
        self.encoded_data = []

        self.build_vocabulary(data)
        self.tokenize_data(data)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def build_vocabulary(self, data):
        unique_chars = set(char for sequence in data for char in sequence)
        for i, char in enumerate(unique_chars, start=2):
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
        self, text_data, image_data, vqvae, quantized_image_shape, verbose=False
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

    def preprocess_data(self):
        encoded_data = []
        for text, image in zip(self.text_data, self.image_data):
            # Tokenize text
            text_tokens = [self.token_to_id[word] for word in text.split()]
            # Quantize image and adjust tokens
            image_tokens = self.vqvae.quantize(image[np.newaxis, ...]).flatten()
            image_tokens += self.image_token_offset

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
    use_cache=True,
):
    model.eval()
    if use_cache:
        model.clear_cache()

    # Initialize samples with <bos> token
    samples = torch.full(
        (num_samples, 1), fill_value=bos_token, dtype=torch.long, device=device
    )

    # Tracks the number of tokens sampled for the current modality
    tokens_sampled_per_modality = torch.zeros(
        num_samples, dtype=torch.long, device=device
    )

    # Initialize modality for each sample, True for text, False for image
    current_modality = torch.full((num_samples,), True, dtype=torch.bool, device=device)

    time_list = []

    for step in range(seq_len):
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
    tokenizer = ColoredImageTokenizer(4)
    train_data = tokenizer.encode(train_data)
    test_data = tokenizer.encode(test_data)

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
    fill_tensor = torch.full((train_data.shape[0], 1), BOS_TOKEN)
    train_data = torch.cat((fill_tensor, train_data), dim=1)

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

    # TODO
    if generate:
        if verbose:
            print("Sampling...")
        num_samples = 9
        samples, _ = multimodal_sample(
            model,
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
        samples_unconditioned = train_dataset.decode(samples)
        # TODO: generate conditioned samples properly
        samples_text_conditioned = samples_image_conditioned = samples_unconditioned
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

    if question == 3:
        if part == "c":
            print(f"Q 3b ds {dataset}")
            model = q3ab_save_results(dataset, "b", q3_b, generate=False, save=False)[
                -1
            ]
            print(f"Q 3c ds {dataset}")
            q3c_save_results(dataset, q3_c, model)
            return
    elif question == 4:
        if part == "b":
            print(f"Q 4b ds {dataset}")
            q4b_save_results(dataset, q4_b)
            return
    elif question == 6:
        if part == "a":
            print("Q 6a")
            q6a_save_results(q6_a)
            return

    raise NotImplementedError(f"Question {question} part {part} not implemented")


if __name__ == "__main__":
    main()
