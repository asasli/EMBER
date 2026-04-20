"""Torch-based anomaly models and feature-learning helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return DEFAULT_DEVICE
    return torch.device(device)


def _align_spec_bank(specs: Sequence[np.ndarray]) -> np.ndarray:
    if not specs:
        raise ValueError("specs must not be empty.")

    min_f = min(np.asarray(spec).shape[0] for spec in specs)
    min_t = min(np.asarray(spec).shape[1] for spec in specs)
    return np.stack(
        [np.asarray(spec, dtype=np.float32)[:min_f, :min_t] for spec in specs],
        axis=0,
    )


class SpectrogramAE(nn.Module):
    """Convolutional autoencoder trained on noise-only PSP spectrograms."""

    def __init__(self, h: int = 64, w: int = 64, latent_dim: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((h, w)),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def prepare_specs_tensor(
    specs_list: Sequence[np.ndarray],
    *,
    h: int = 64,
    w: int = 64,
    noise_mean: torch.Tensor | None = None,
    noise_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize spectrograms and normalize them with the provided or inferred stats."""

    tensors = []
    for spec in specs_list:
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        tensors.append(tensor.squeeze(0))
    specs_t = torch.stack(tensors)

    if noise_mean is None:
        noise_mean = specs_t.mean()
    if noise_std is None:
        noise_std = specs_t.std() + 1e-6

    return (specs_t - noise_mean) / noise_std, noise_mean, noise_std


def train_autoencoder(
    noise_specs_np: Sequence[np.ndarray],
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 8,
    latent_dim: int = 24,
    h: int = 64,
    w: int = 64,
    validation_fraction: float = 0.0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best: bool = True,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[SpectrogramAE, torch.Tensor, torch.Tensor]:
    """Train the spectrogram autoencoder on the noise-only subset.

    When ``validation_fraction`` is positive, training monitors a held-out
    validation split and optionally uses early stopping. In all cases the best
    checkpoint on the monitored loss can be restored before returning.
    """

    device = _resolve_device(device)
    specs_t, mu, std = prepare_specs_tensor(noise_specs_np, h=h, w=w)
    if len(specs_t) == 0:
        raise ValueError("noise_specs_np must not be empty.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when provided.")

    ae = SpectrogramAE(h=h, w=w, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_specs = len(specs_t)
    if validation_fraction > 0.0 and n_specs > 1:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_specs)
        n_val = int(round(validation_fraction * n_specs))
        n_val = min(max(1, n_val), n_specs - 1)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        train_idx = np.arange(n_specs)
        val_idx = np.asarray([], dtype=int)

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    X_train = specs_t[train_idx].to(device)
    X_val = specs_t[val_idx].to(device) if len(val_idx) else None

    for epoch in range(epochs):
        ae.train()
        idx = torch.randperm(len(X_train), device=device)
        total_loss = 0.0

        for start in range(0, len(X_train), batch_size):
            batch = X_train[idx[start : start + batch_size]]
            opt.zero_grad()
            recon, z = ae(batch)

            loss_recon = F.mse_loss(recon, batch)
            loss_reg = 0.001 * z.pow(2).mean()
            loss = loss_recon + loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
            total_loss += float(loss_recon.item()) * len(batch)

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(X_train))

        monitor_loss = avg_train_loss
        avg_val_loss: float | None = None
        if X_val is not None and len(X_val) > 0:
            ae.eval()
            with torch.no_grad():
                val_total_loss = 0.0
                for start in range(0, len(X_val), batch_size):
                    batch = X_val[start : start + batch_size]
                    recon, _ = ae(batch)
                    val_total_loss += float(F.mse_loss(recon, batch).item()) * len(batch)
            avg_val_loss = val_total_loss / max(1, len(X_val))
            monitor_loss = avg_val_loss

        if monitor_loss < best_loss - early_stopping_min_delta:
            best_loss = monitor_loss
            best_state = {name: value.detach().cpu().clone() for name, value in ae.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 50 == 0:
            if avg_val_loss is None:
                print(f"  Epoch {epoch + 1:3d}/{epochs} | Train loss: {avg_train_loss:.4f}")
            else:
                print(
                    f"  Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}"
                )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1:3d}/{epochs} | Best monitored loss: {best_loss:.4f}")
            break

    if restore_best and best_state is not None:
        ae.load_state_dict(best_state)
    return ae, mu, std


def score_with_ae(
    ae: SpectrogramAE,
    all_specs_np: Sequence[np.ndarray],
    mu: torch.Tensor,
    std: torch.Tensor,
    *,
    h: int = 64,
    w: int = 64,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the full spectrogram bank with reconstruction error."""

    device = _resolve_device(device)
    specs_t, _, _ = prepare_specs_tensor(all_specs_np, h=h, w=w, noise_mean=mu, noise_std=std)

    ae.eval()
    scores = []
    latents = []
    with torch.no_grad():
        for start in range(0, len(specs_t), batch_size):
            batch = specs_t[start : start + batch_size].to(device)
            recon, z = ae(batch)
            err = F.mse_loss(recon, batch, reduction="none").mean(dim=(1, 2, 3))
            scores.append(err.cpu().numpy())
            latents.append(z.cpu().numpy())

    return np.concatenate(scores), np.vstack(latents)


def score_with_ae_loo(
    all_specs_np: Sequence[np.ndarray],
    noise_idx: Sequence[int],
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 8,
    latent_dim: int = 24,
    h: int = 64,
    w: int = 64,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each sample with an autoencoder trained without that sample.

    Noise points are scored with a leave-one-noise-out autoencoder. Anomaly
    points are scored with an autoencoder trained on the full noise bank,
    which keeps the evaluation honest without retraining a separate model for
    every anomaly.
    """

    specs = [np.asarray(spec, dtype=np.float32) for spec in all_specs_np]
    noise_idx = np.asarray(noise_idx, dtype=int)
    if len(specs) == 0:
        raise ValueError("all_specs_np must not be empty.")
    if len(noise_idx) < 2:
        raise ValueError("score_with_ae_loo requires at least two noise samples.")

    n_samples = len(specs)
    noise_set = set(map(int, noise_idx))
    signal_idx = [idx for idx in range(n_samples) if idx not in noise_set]
    scores = np.zeros(n_samples, dtype=np.float32)
    latents: np.ndarray | None = None

    def _fit_on_noise(train_noise_idx: np.ndarray, note: str) -> tuple[SpectrogramAE, torch.Tensor, torch.Tensor]:
        if len(train_noise_idx) == 0:
            raise ValueError("Each AE fold needs at least one training noise sample.")
        if verbose:
            print(f"Training strict AE fold for {note} on {len(train_noise_idx)} noise samples...")
        return train_autoencoder(
            [specs[idx] for idx in train_noise_idx],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            latent_dim=latent_dim,
            h=h,
            w=w,
            device=device,
            verbose=False,
        )

    if signal_idx:
        ae_full, mu_full, std_full = _fit_on_noise(noise_idx, "all anomaly samples")
        signal_scores, signal_latents = score_with_ae(
            ae_full,
            [specs[idx] for idx in signal_idx],
            mu_full,
            std_full,
            h=h,
            w=w,
            batch_size=batch_size,
            device=device,
        )
        scores[np.asarray(signal_idx, dtype=int)] = signal_scores.astype(np.float32)
        latents = np.zeros((n_samples, signal_latents.shape[1]), dtype=np.float32)
        latents[np.asarray(signal_idx, dtype=int)] = signal_latents.astype(np.float32)

    for fold_idx, test_i in enumerate(noise_idx, start=1):
        train_noise_idx = noise_idx[noise_idx != test_i]
        ae_fold, mu_fold, std_fold = _fit_on_noise(
            train_noise_idx,
            f"noise sample {fold_idx}/{len(noise_idx)} (held out index {int(test_i)})",
        )
        fold_scores, fold_latents = score_with_ae(
            ae_fold,
            [specs[int(test_i)]],
            mu_fold,
            std_fold,
            h=h,
            w=w,
            batch_size=1,
            device=device,
        )
        scores[int(test_i)] = float(fold_scores[0])
        if latents is None:
            latents = np.zeros((n_samples, fold_latents.shape[1]), dtype=np.float32)
        latents[int(test_i)] = fold_latents[0].astype(np.float32)

    if latents is None:
        latents = np.zeros((n_samples, latent_dim), dtype=np.float32)

    return scores, latents


def build_noise_dataloaders(
    noise_specs: Sequence[np.ndarray],
    *,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Align the noise bank and return notebook-style train/val loaders."""

    aligned = _align_spec_bank(noise_specs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(aligned))
    aligned = aligned[perm]

    split = max(1, int(train_fraction * len(aligned)))
    train_np = aligned[:split]
    val_np = aligned[split:]
    if len(val_np) == 0:
        val_np = train_np[: max(1, min(4, len(train_np)))]

    train_x = torch.tensor(train_np, dtype=torch.float32).unsqueeze(1)
    val_x = torch.tensor(val_np, dtype=torch.float32).unsqueeze(1)

    return (
        DataLoader(train_x, batch_size=batch_size, shuffle=True),
        DataLoader(val_x, batch_size=batch_size, shuffle=False),
        train_x,
        val_x,
    )


class AffineCoupling(nn.Module):
    """Simple affine coupling block for the notebook's toy flow model."""

    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("AffineCoupling requires an even number of channels.")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)

        if not reverse:
            z2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2, 3])
        else:
            z2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2, 3])

        z = torch.cat([x1, z2], dim=1)
        return z, log_det


class SimpleFlow(nn.Module):
    """Notebook-style stack of affine coupling layers."""

    def __init__(self, in_channels: int = 2, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(in_channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = 0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            z, _ = layer(z, reverse=True)
        return z


def log_prob(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """Compute the standard-normal flow log probability."""

    log_two_pi = torch.log(torch.tensor(2 * np.pi, dtype=z.dtype, device=z.device))
    log_pz = -0.5 * torch.sum(z**2 + log_two_pi, dim=[1, 2, 3])
    return log_pz + log_det


def anomaly_score(model: SimpleFlow, x: torch.Tensor) -> torch.Tensor:
    """Return the negative flow log likelihood so larger means more anomalous."""

    model.eval()
    with torch.no_grad():
        if x.shape[1] == 1:
            x = torch.cat([x, x], dim=1)
        x = (x - x.mean()) / (x.std() + 1e-6)
        z, log_det = model(x)
        return -log_prob(z, log_det)


def train_simple_flow(
    noise_specs: Sequence[np.ndarray],
    *,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> tuple[SimpleFlow, dict[str, object]]:
    """Train the lightweight coupling flow on the aligned noise bank."""

    del seed
    device = _resolve_device(device)
    train_loader, val_loader, train_x, val_x = build_noise_dataloaders(
        noise_specs,
        batch_size=batch_size,
    )

    model = SimpleFlow(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for x in train_loader:
            x = x.to(device)
            if x.shape[1] == 1:
                x = torch.cat([x, x], dim=1)
            x = (x - x.mean()) / (x.std() + 1e-6)

            z, log_det = model(x)
            loss = -log_prob(z, log_det).mean()
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch}: loss={mean_loss:.4f}")

    return model, {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_shape": tuple(train_x.shape),
        "val_shape": tuple(val_x.shape),
    }


def score_simple_flow(
    model: SimpleFlow,
    specs: Sequence[np.ndarray],
    *,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Score a bank of spectrograms with the trained lightweight flow."""

    device = _resolve_device(device)
    aligned = _align_spec_bank(specs)
    tensor_bank = torch.tensor(aligned, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(tensor_bank, batch_size=batch_size, shuffle=False)

    scores = []
    for batch in loader:
        batch_scores = anomaly_score(model, batch.to(device))
        scores.append(batch_scores.detach().cpu().numpy())
    return np.concatenate(scores)


class ContrastiveEncoder(nn.Module):
    """Small CNN encoder with a projection head for contrastive learning."""

    def __init__(self, in_ch: int = 1, latent_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze(-1).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Compatibility loss mirroring the notebook helper."""

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    labels = (labels + z.size(0) // 2) % z.size(0)
    return F.cross_entropy(sim, labels)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Create a second spectrogram view with light amplitude/noise/shift augmentations."""

    y = x.clone()
    scale = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device)
    y = y * scale
    y = y + 0.03 * torch.randn_like(y)

    if y.size(-1) > 8:
        shifts = torch.randint(-6, 7, (x.size(0),), device=x.device)
        for i, shift in enumerate(shifts.tolist()):
            y[i] = torch.roll(y[i], shifts=shift, dims=-1)

    if y.size(-2) > 8 and torch.rand(1, device=x.device).item() < 0.5:
        freq_shift = int(torch.randint(-4, 5, (1,), device=x.device).item())
        y = torch.roll(y, shifts=freq_shift, dims=-2)

    return y


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE loss used by the contrastive encoder notebook cell."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    batch_size = z1.size(0)
    eye = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    targets = torch.arange(batch_size, 2 * batch_size, device=z.device)
    targets = torch.cat([targets, torch.arange(0, batch_size, device=z.device)])
    return F.cross_entropy(sim, targets)


def encode_specs(
    model: ContrastiveEncoder,
    specs_list: Sequence[np.ndarray],
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Encode each spectrogram into the contrastive backbone feature space."""

    device = _resolve_device(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for spec in specs_list:
            batch_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            hidden = model.encode(batch_tensor)
            embeddings.append(hidden.squeeze(0).cpu().numpy())
    return np.vstack(embeddings)


def train_contrastive_encoder(
    noise_specs: Sequence[np.ndarray],
    *,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    latent_dim: int = 128,
    projection_dim: int = 64,
    device: str | torch.device | None = None,
) -> ContrastiveEncoder:
    """Train the notebook-style contrastive encoder on the aligned noise bank."""

    device = _resolve_device(device)
    train_loader, _, _, _ = build_noise_dataloaders(noise_specs, batch_size=batch_size)
    model = ContrastiveEncoder(in_ch=1, latent_dim=latent_dim, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for x in train_loader:
            x = x.to(device)
            x1 = augment_batch(x)
            x2 = augment_batch(x)

            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2, temperature=0.2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        print(f"Epoch {epoch}: contrastive loss={np.mean(epoch_losses):.4f}")

    return model


def fit_zuko_maf(
    train_features: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    transforms: int = 5,
    hidden_features: Sequence[int] = (128, 128),
    device: str | torch.device | None = None,
):
    """Train the zuko MAF that appears in the later notebook experiments."""

    try:
        import zuko
    except ImportError as exc:
        raise ImportError(
            "fit_zuko_maf requires zuko. Install it manually if you want the notebook's MAF stage."
        ) from exc

    device = _resolve_device(device)
    flow_dim = int(np.asarray(train_features).shape[1])
    flow = zuko.flows.MAF(
        features=flow_dim,
        transforms=transforms,
        hidden_features=list(hidden_features),
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    X = torch.tensor(np.asarray(train_features), dtype=torch.float32).to(device)

    for _ in range(epochs):
        loss = -flow().log_prob(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return flow


def flow_score(
    flow,
    X: np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Score features with a trained zuko-style flow object."""

    device = _resolve_device(device)
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        return -flow().log_prob(X_t).cpu().numpy()


def blend_scores(
    score_arrays: Sequence[np.ndarray],
    *,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Blend multiple score vectors into one weighted anomaly score."""

    arrays = [np.asarray(scores, dtype=float) for scores in score_arrays]
    if not arrays:
        raise ValueError("score_arrays must not be empty.")

    if weights is None:
        weights_arr = np.ones(len(arrays), dtype=float) / len(arrays)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()

    return np.tensordot(weights_arr, np.vstack(arrays), axes=1)


# ---------------------------------------------------------------------------
# β-VAE (Variational Autoencoder)
# ---------------------------------------------------------------------------

class SpectrogramVAE(nn.Module):
    """β-Variational Autoencoder for 2-D spectrograms.

    Anomaly score = ELBO loss = reconstruction_loss + beta * KL divergence.
    Trained on background-only data; anomalies produce high ELBO scores.

    Parameters
    ----------
    h, w : int
        Spectrogram height and width after resizing.
    latent_dim : int
        Dimension of the latent code ``z``.
    beta : float
        KL weight (> 1 encourages disentanglement; < 1 gives more expressive
        reconstruction at the cost of regularisation).
    """

    def __init__(self, h: int = 64, w: int = 64, latent_dim: int = 32, beta: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4), nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((h, w)),
            nn.Conv2d(16, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, mu, logvar)``."""
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def elbo_loss(self, x: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """Per-sample ELBO = recon_loss + beta * KL.  Higher = more anomalous."""
        if beta is None:
            beta = self.beta
        recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
        kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return recon_loss + beta * kl_loss


def train_vae(
    noise_specs_np: Sequence[np.ndarray],
    *,
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 16,
    latent_dim: int = 32,
    beta: float = 0.5,
    h: int = 64,
    w: int = 64,
    val_frac: float = 0.15,
    patience: int = 25,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple["SpectrogramVAE", torch.Tensor, torch.Tensor]:
    """Train a β-VAE on the background-only (noise) subset with early stopping.

    Parameters
    ----------
    noise_specs_np : list of (H, W) arrays
        Background-only training spectrograms.
    epochs : int
        Maximum training epochs (default 400).
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    latent_dim : int
        VAE latent dimension.
    beta : float
        KL weight for the ELBO.
    h, w : int
        Resize target for spectrograms.
    val_frac : float
        Fraction held out for early-stopping validation (default 0.15).
    patience : int
        Stop if val loss does not improve for this many epochs (default 25).
    seed : int, optional
        Random seed for reproducibility.
    device : str or None
        PyTorch device.
    verbose : bool
        Print epoch losses.

    Returns
    -------
    (model, noise_mu, noise_std)
        Fitted ``SpectrogramVAE`` and the normalisation statistics used.
    """
    device = _resolve_device(device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    specs_t, mu_norm, std_norm = prepare_specs_tensor(noise_specs_np, h=h, w=w)

    # Train / validation split for early stopping
    N = len(specs_t)
    n_val = max(1, int(np.floor(val_frac * N)))
    perm = torch.randperm(N)
    X_train = specs_t[perm[n_val:]].to(device)
    X_val   = specs_t[perm[:n_val]].to(device)

    model = SpectrogramVAE(h=h, w=w, latent_dim=latent_dim, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(5, patience // 5)
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0
    log_every = max(1, epochs // 10)

    for epoch in range(epochs):
        # Training pass
        model.train()
        shuf = torch.randperm(len(X_train), device=device)
        train_total, train_n = 0.0, 0
        for s in range(0, len(X_train), batch_size):
            batch = X_train[shuf[s: s + batch_size]]
            opt.zero_grad()
            loss = model.elbo_loss(batch, beta=beta).mean()
            loss.backward()
            opt.step()
            train_total += float(loss.detach()) * len(batch)
            train_n += len(batch)

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_loss = float(model.elbo_loss(X_val, beta=beta).mean())
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1 or improved):
            marker = " *" if improved else f" ({no_improve}/{patience})"
            print(
                f"  VAE {epoch + 1:4d}/{epochs}"
                f"  train={train_total / train_n:.4f}"
                f"  val={val_loss:.4f}"
                f"  best={best_val_loss:.4f}{marker}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1}  best_val={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, mu_norm, std_norm


def score_vae(
    model: "SpectrogramVAE",
    specs: Sequence[np.ndarray],
    mu_norm: torch.Tensor,
    std_norm: torch.Tensor,
    *,
    h: int = 64,
    w: int = 64,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score spectrograms with a trained β-VAE.

    Parameters
    ----------
    model : SpectrogramVAE
        Fitted VAE model.
    specs : list of (H, W) arrays
        Spectrograms to score (background + anomaly).
    mu_norm, std_norm : Tensor
        Normalisation statistics from ``train_vae``.
    h, w : int
        Resize target matching training.
    batch_size : int
        Inference batch size.
    device : str or None
        PyTorch device.

    Returns
    -------
    (elbo_scores, latent_vectors)
        ``elbo_scores`` shape (N,) — higher = more anomalous.
        ``latent_vectors`` shape (N, latent_dim) — ``z_mu`` for each sample.
    """
    device  = _resolve_device(device)
    specs_t, _, _ = prepare_specs_tensor(specs, h=h, w=w,
                                          noise_mean=mu_norm, noise_std=std_norm)
    X = specs_t.to(device)
    model.eval().to(device)

    all_scores  = []
    all_latents = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            scores = model.elbo_loss(batch)
            mu, _  = model.encode(batch)
            all_scores.append(scores.cpu().numpy())
            all_latents.append(mu.cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_latents)
