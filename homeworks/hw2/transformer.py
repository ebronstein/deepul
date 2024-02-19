import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
