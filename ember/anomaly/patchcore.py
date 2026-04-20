"""PatchCore-style anomaly detector using a frozen pretrained CNN backbone.

Implements the core idea of Roth et al. (2022) "Towards Total Recall in
Industrial Anomaly Detection" at reduced spatial resolution for fast inference:

1. Frozen WideResNet-50 (ImageNet weights) extracts features from layers 2 & 3.
2. Features are pooled to a ``pool_size × pool_size`` spatial grid (default 4×4 = 16 patches).
3. A KNN index is built on the flattened (N×16, 1536) background training patches.
4. Anomaly score = max over patches of the mean-k-NN distance to the memory bank.

No fine-tuning is performed — ImageNet weights transfer well to scientific images.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

import numpy as np


def _load_torchvision():
    """Import torchvision, working around libstdc++ version mismatches on some HPC clusters."""
    try:
        import torchvision.models as tv
        return tv
    except ImportError:
        pass

    # Attempt: preload the conda env's newer libstdc++ before PIL's C extension loads
    import glob, os
    conda_prefix = os.environ.get(
        "CONDA_PREFIX",
        str(__file__).split("lib/python")[0] if "lib/python" in __file__ else "",
    )
    for pattern in [
        f"{conda_prefix}/lib/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    ]:
        for path in glob.glob(pattern):
            try:
                ctypes.CDLL(path)
                import torchvision.models as tv
                return tv
            except Exception:
                continue

    raise ImportError(
        "torchvision could not be imported. "
        "Install with `pip install torchvision` or `pip install -e .[anomaly]`."
    )


class PatchCoreDetector:
    """Patch-level anomaly detector backed by a frozen WideResNet-50.

    Parameters
    ----------
    pool_size : int
        Spatial output size of AdaptiveAvgPool2d applied to backbone feature
        maps.  Gives ``pool_size²`` patches per spectrogram (default: 4 → 16).
    knn_k : int
        Number of nearest neighbours for scoring.
    img_size : int
        Spectrograms are resized to ``(img_size, img_size)`` before the backbone.
    device : str or None
        PyTorch device (``"cuda"``, ``"cpu"``, or ``None`` for auto-detect).
    batch_size : int
        Number of spectrograms processed in one forward pass.
    """

    def __init__(
        self,
        pool_size: int = 4,
        knn_k: int = 5,
        img_size: int = 224,
        device: str | None = None,
        batch_size: int = 16,
        n_jobs: int = 1,
    ):
        self.pool_size  = pool_size
        self.knn_k      = knn_k
        self.img_size   = img_size
        self.batch_size = batch_size
        self.n_jobs     = n_jobs
        self._device    = None
        self._backbone  = None
        self._knn       = None
        self._device_str = device

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_device(self):
        if self._device is None:
            import torch
            dev = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev)
        return self._device

    def _get_backbone(self):
        if self._backbone is not None:
            return self._backbone

        import torch
        import torch.nn as nn

        tv = _load_torchvision()
        device = self._get_device()

        backbone = tv.wide_resnet50_2(weights=tv.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        backbone.eval().to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)

        feat_cache: dict[str, "torch.Tensor"] = {}

        def _hook(name):
            def fn(m, inp, out):
                feat_cache[name] = out.detach()
            return fn

        backbone.layer2.register_forward_hook(_hook("l2"))
        backbone.layer3.register_forward_hook(_hook("l3"))

        pool = nn.AdaptiveAvgPool2d(self.pool_size).to(device)

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self._backbone    = backbone
        self._feat_cache  = feat_cache
        self._pool        = pool
        self._im_mean     = imagenet_mean
        self._im_std      = imagenet_std
        return backbone

    def _preprocess(self, specs: Sequence[np.ndarray]):
        import torch
        import torch.nn.functional as F

        device = self._get_device()
        tensors = []
        for s in specs:
            t = torch.from_numpy(np.asarray(s, dtype=np.float32))
            lo, hi = t.min(), t.max()
            t = (t - lo) / (hi - lo + 1e-9)
            t = t.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).clone()
            t = F.interpolate(t, (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            tensors.append(t)
        x = torch.cat(tensors, 0).to(device)
        return (x - self._im_mean) / self._im_std

    def _extract_patch_features(
        self, specs: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Return (N, pool_size², 1536) float32 patch features."""
        import torch

        backbone = self._get_backbone()
        device   = self._get_device()
        all_patches = []

        for i in range(0, len(specs), self.batch_size):
            batch = specs[i : i + self.batch_size]
            x = self._preprocess(batch)
            with torch.no_grad():
                _ = backbone(x)
            l2 = self._pool(self._feat_cache["l2"])   # (B, 512, P, P)
            l3 = self._pool(self._feat_cache["l3"])   # (B, 1024, P, P)
            combined = torch.cat([l2, l3], dim=1)      # (B, 1536, P, P)
            B, C, P, _ = combined.shape
            patches = combined.permute(0, 2, 3, 1).reshape(B, P * P, C)
            all_patches.append(patches.cpu().numpy().astype(np.float32))

        return np.concatenate(all_patches, axis=0)   # (N, P², 1536)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, bg_train_specs: Sequence[np.ndarray]) -> "PatchCoreDetector":
        """Build the patch memory bank from background training spectrograms.

        Parameters
        ----------
        bg_train_specs : list of (H, W) arrays
            Background-only training spectrograms.
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "PatchCoreDetector requires scikit-learn. "
                "Install with `pip install -e .[anomaly]`."
            ) from exc

        patches = self._extract_patch_features(bg_train_specs)  # (N, P², 1536)
        mem_bank = patches.reshape(-1, patches.shape[-1])         # (N*P², 1536)

        knn = NearestNeighbors(
            n_neighbors=self.knn_k,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=self.n_jobs,
        )
        knn.fit(mem_bank)
        self._knn = knn
        return self

    def score(self, specs: Sequence[np.ndarray]) -> np.ndarray:
        """Score spectrograms — higher means more anomalous.

        Score = max over patches of mean-k-NN distance to the memory bank.

        Parameters
        ----------
        specs : list of (H, W) arrays
            Spectrograms to score.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        if self._knn is None:
            raise RuntimeError("Call fit() before score().")

        patches = self._extract_patch_features(specs)  # (N, P², 1536)
        N, P2, C = patches.shape
        flat = patches.reshape(N * P2, C)

        dists, _ = self._knn.kneighbors(flat)      # (N*P², k)
        nn_score  = dists.mean(axis=1)              # mean-k dist per patch
        per_img   = nn_score.reshape(N, P2).max(axis=1)  # worst patch per image
        return per_img
