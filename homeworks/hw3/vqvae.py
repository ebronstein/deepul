"""VQ-VAE implementation."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformer
from torch.utils import data


def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))

    losses = OrderedDict()
    for x in train_loader:
        x = x.cuda()
        optimizer.zero_grad()
        out = model.loss(x)
        out["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f"Epoch {epoch}"
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f", {k} {avg_loss:.4f}"

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = "Test "
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f", {k} {total_losses[k]:.4f}"
        if not quiet:
            print(desc)
    return total_losses


def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    epochs, lr = train_args["epochs"], train_args["lr"]
    grad_clip = train_args.get("grad_clip", None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return train_losses, test_losses


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Codebook(nn.Module):

    def __init__(self, codebook_size, code_dim):
        super().__init__()
        self.code_dim = code_dim
        self.codebook_size = codebook_size

        self.embedding = nn.Embedding(codebook_size, code_dim)
        # Initialize embedding to uniform random between -1/codebook_size and 1/codebook_size
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        flattened_z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        weight = self.embedding.weight
        # Compute distances between z and embedding vectors
        distances = (
            (flattened_z**2).sum(dim=1, keepdim=True)
            - 2 * torch.mm(flattened_z, weight.t())
            + (weight.t() ** 2).sum(dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        batch_size, _, height, width = z.shape
        encoding_indices = encoding_indices.view(batch_size, height, width)
        code = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()
        detached_code = (code - z).detach() + z
        return code, detached_code, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, code_dim, code_size, decoder_tanh=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, code_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(code_dim),
            nn.Conv2d(code_dim, code_dim, 4, stride=2, padding=1),
            ResBlock(code_dim),
            ResBlock(code_dim),
        )

        self.code_size = code_size
        self.codebook = Codebook(code_size, code_dim)

        decoder_layers = [
            ResBlock(code_dim),
            ResBlock(code_dim),
            nn.ReLU(),
            nn.BatchNorm2d(code_dim),
            nn.ConvTranspose2d(code_dim, code_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(code_dim),
            nn.ConvTranspose2d(code_dim, 3, 4, stride=2, padding=1),
        ]
        if decoder_tanh:
            decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            return self.codebook(z)[2]

    def decode(self, z):
        with torch.no_grad():
            code = self.codebook.embedding(z).permute(0, 3, 1, 2).contiguous()
            return self.decoder(code).permute(0, 2, 3, 1).cpu().numpy()

    def reconstruct(self, x):
        # Encode
        z = self.encoder(x)
        # Get the code
        code, code_stop_grad, _ = self.codebook(z)
        # Decode the code token
        x_recon = self.decoder(code_stop_grad)
        return x_recon

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        # Get the code
        code, code_stop_grad, _ = self.codebook(z)
        # Decode the code token
        x_recon = self.decoder(code_stop_grad)

        # Commitment loss
        commitment_loss = torch.mean((z - code.detach()) ** 2)
        # Embedding loss
        embedding_loss = torch.mean((code - z.detach()) ** 2)
        # Total regularization loss
        reg_loss = commitment_loss + embedding_loss

        return x_recon, reg_loss

    def loss(self, x):
        x_recon, reg_loss = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + reg_loss
        return OrderedDict(loss=total_loss, recon_loss=recon_loss, reg_loss=reg_loss)


def make_dataset_for_prior(vqvae, data_loader):
    data = []
    with torch.no_grad():
        for x in data_loader:
            z = vqvae.encode_code(x.cuda())
            data.append(z.long())
    return torch.cat(data, dim=0)


def q3(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """
    epochs = 20
    lr = 1e-3
    grad_clip = 1
    batch_size = 128
    code_dim, code_size = 256, 128

    # Preprocess data
    train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255 * 2 - 1).astype(
        "float32"
    )
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255 * 2 - 1).astype("float32")
    assert train_data.min() >= -1 and train_data.max() <= 1
    assert test_data.min() >= -1 and test_data.max() <= 1

    # Train VQ-VAE
    vqvae = VQVAE(code_dim, code_size).cuda()
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    train_losses, test_losses = train_epochs(
        vqvae,
        train_loader,
        test_loader,
        dict(epochs=epochs, lr=lr, grad_clip=grad_clip),
        quiet=False,
    )
    vqvae_train_losses, vqvae_test_losses = train_losses["loss"], test_losses["loss"]

    # Create prior dataset
    prior_train_data, prior_test_data = make_dataset_for_prior(
        vqvae, train_loader
    ), make_dataset_for_prior(vqvae, test_loader)
    prior_train_data = prior_train_data.cpu().numpy()
    prior_test_data = prior_test_data.cpu().numpy()
    # Flatten the data to 1D
    prior_train_data = prior_train_data.reshape(prior_train_data.shape[0], -1)
    prior_test_data = prior_test_data.reshape(prior_test_data.shape[0], -1)
    bos_token = code_size
    # Append the beginning of sentence token to the data
    prior_train_data = np.concatenate(
        [np.full((prior_train_data.shape[0], 1), bos_token), prior_train_data], axis=1
    )
    prior_test_data = np.concatenate(
        [np.full((prior_test_data.shape[0], 1), bos_token), prior_test_data], axis=1
    )
    # Create dataloaders
    prior_train_loader = data.DataLoader(
        prior_train_data, batch_size=batch_size, shuffle=True
    )
    prior_test_loader = data.DataLoader(prior_test_data, batch_size=batch_size)

    # Train Transformer prior
    seq_len = prior_train_data.shape[1]
    vocab_size = code_size + 1
    num_layers = 4
    d_model = 128
    num_heads = 4
    dropout = 0.0
    prior = transformer.Transformer(
        seq_len, vocab_size, num_layers, d_model, num_heads, dropout
    ).cuda()

    prior_epochs = 15
    prior_train_losses, prior_test_losses = transformer.train_transformer(
        prior,
        prior_train_loader,
        prior_test_loader,
        epochs=prior_epochs,
        lr=1e-3,
        device="cuda",
        verbose=True,
    )

    # Sample
    samples, _ = transformer.sample(prior, 100, seq_len - 1, "cuda", bos_token)
    samples = torch.from_numpy(samples).long().cuda()
    samples = samples.reshape(-1, 8, 8)
    samples = (vqvae.decode(samples) * 0.5 + 0.5) * 255

    # Reconstructions
    original = next(iter(test_loader))[:50].cuda()
    with torch.no_grad():
        z = vqvae.encode(original)
        recon = vqvae.decode(z)
    original = original.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = np.stack((original, recon), axis=1).reshape((-1, 32, 32, 3))
    reconstructions = (reconstructions * 0.5 + 0.5) * 255

    return (
        vqvae_train_losses,
        vqvae_test_losses,
        prior_train_losses,
        prior_test_losses,
        samples,
        reconstructions,
    )
