"""Pretrained vision-backbone helpers for anomaly-investigation notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ember.anomaly.neural import DEFAULT_DEVICE, prepare_specs_tensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONES = ("resnet18", "resnet50", "convnext", "dinov2", "clip")
BACKBONE_INPUT_SIZES = {
    "resnet18": (224, 224),
    "resnet50": (224, 224),
    "convnext": (224, 224),
    "dinov2": (518, 518),
    "clip": (224, 224),
}


@dataclass(frozen=True)
class PreparedSpectrogramBank:
    """Resized and noise-normalized spectrogram bank for backbone extraction."""

    specs: np.ndarray
    noise_mean: float
    noise_std: float
    target_shape: tuple[int, int]


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return DEFAULT_DEVICE
    return torch.device(device)


def prepare_multimodel_spectrograms(
    specs_list: Sequence[np.ndarray],
    *,
    noise_indices: Sequence[int] | None = None,
    h: int = 224,
    w: int = 224,
) -> PreparedSpectrogramBank:
    """Resize spectrograms and normalize them with noise-subset statistics."""

    if len(specs_list) == 0:
        raise ValueError("specs_list must not be empty.")

    if noise_indices is None:
        noise_specs = list(specs_list)
    else:
        noise_specs = [specs_list[int(idx)] for idx in noise_indices]
        if not noise_specs:
            raise ValueError("noise_indices must reference at least one spectrogram.")

    _, noise_mean, noise_std = prepare_specs_tensor(noise_specs, h=h, w=w)
    specs_t, _, _ = prepare_specs_tensor(
        specs_list,
        h=h,
        w=w,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )

    return PreparedSpectrogramBank(
        specs=specs_t.squeeze(1).cpu().numpy(),
        noise_mean=float(noise_mean.item()),
        noise_std=float(noise_std.item()),
        target_shape=(h, w),
    )


def prepare_embedding_spectrograms(
    specs_list: Sequence[np.ndarray],
    *,
    noise_indices: Sequence[int] | None = None,
    h: int = 224,
    w: int = 224,
) -> PreparedSpectrogramBank:
    """Alias for :func:`prepare_multimodel_spectrograms`."""

    return prepare_multimodel_spectrograms(specs_list, noise_indices=noise_indices, h=h, w=w)


class MultiModelExtractor:
    """Extract feature vectors from PSP spectrograms using pretrained backbones."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        model_names: Sequence[str] | None = None,
        strict: bool = False,
        verbose: bool = False,
    ):
        self.device = _resolve_device(device)
        self.requested_model_names = tuple(model_names or DEFAULT_BACKBONES)
        self.strict = bool(strict)
        self.verbose = bool(verbose)
        self.models: dict[str, nn.Module] = {}
        self.load_errors: dict[str, str] = {}
        self._load_requested_models()

    @property
    def loaded_model_names(self) -> list[str]:
        """Return the successfully loaded backbone names."""

        return list(self.models)

    def _load_requested_models(self) -> None:
        unknown = [name for name in self.requested_model_names if name not in BACKBONE_INPUT_SIZES]
        if unknown:
            raise ValueError(f"Unknown backbone(s): {unknown}")

        for model_name in self.requested_model_names:
            try:
                self.models[model_name] = self._load_model(model_name)
            except Exception as exc:
                self.load_errors[model_name] = f"{type(exc).__name__}: {exc}"

        if self.strict and self.load_errors:
            missing = ", ".join(f"{name} [{msg}]" for name, msg in self.load_errors.items())
            raise RuntimeError(f"Failed to load requested backbone(s): {missing}")

        if self.verbose:
            loaded = self.loaded_model_names or ["none"]
            print(f"Loaded backbones: {loaded}")
            if self.load_errors:
                print(f"Skipped backbones: {self.load_errors}")

    def _load_model(self, model_name: str) -> nn.Module:
        if model_name in {"resnet18", "resnet50"}:
            try:
                import torchvision.models as tv_models
            except ImportError as exc:
                raise ImportError(
                    "Torchvision is required for ResNet backbones. "
                    "Install `pip install -e .[anomaly-notebook]`."
                ) from exc

            if model_name == "resnet18":
                weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
                model = tv_models.resnet18(weights=weights)
            else:
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
                model = tv_models.resnet50(weights=weights)
            return nn.Sequential(*list(model.children())[:-1]).eval().to(self.device)

        if model_name in {"convnext", "dinov2"}:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "timm is required for ConvNeXt and DINOv2 backbones. "
                    "Install `pip install -e .[anomaly-notebook]`."
                ) from exc

            model_id = (
                "convnext_tiny.fb_in22k_ft_in1k"
                if model_name == "convnext"
                else "vit_small_patch14_dinov2.lvd142m"
            )
            return timm.create_model(model_id, pretrained=True, num_classes=0).eval().to(self.device)

        if model_name == "clip":
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError(
                    "open_clip is required for the CLIP backbone. "
                    "Install `pip install -e .[anomaly-notebook]`."
                ) from exc

            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            return model.visual.eval().to(self.device)

        raise ValueError(f"Unsupported backbone: {model_name}")

    def _prep(self, specs: np.ndarray, model_name: str) -> torch.Tensor:
        array = np.asarray(specs, dtype=np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, np.newaxis, :, :]
        elif array.ndim == 3:
            array = array[:, np.newaxis, :, :]
        elif array.ndim != 4:
            raise ValueError("Expected a spectrogram batch with shape (n, h, w) or (n, c, h, w).")

        tensor = torch.tensor(array, dtype=torch.float32)
        target_h, target_w = BACKBONE_INPUT_SIZES[model_name]
        if tensor.shape[-2:] != (target_h, target_w):
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] != 3:
            raise ValueError("Expected 1 or 3 channels for the prepared spectrogram bank.")

        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        tensor = tensor * 0.2 + 0.5

        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
        return ((tensor - mean) / std).to(self.device)

    @staticmethod
    def _pool_output(output: torch.Tensor | np.ndarray | Mapping[str, object] | Sequence[object]) -> torch.Tensor:
        if isinstance(output, Mapping):
            for key in (
                "pooler_output",
                "last_hidden_state",
                "x_norm_clstoken",
                "image_embeds",
                "logits",
            ):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))

        if isinstance(output, (list, tuple)):
            output = output[0]

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)

        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unsupported backbone output type: {type(output)!r}")

        if output.ndim == 4:
            output = output.mean(dim=(-2, -1))
        elif output.ndim == 3:
            output = output.mean(dim=1)
        elif output.ndim == 1:
            output = output.unsqueeze(0)

        return output

    def extract(
        self,
        specs: np.ndarray,
        *,
        batch_size: int = 32,
        model_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract features for every loaded backbone over the provided bank."""

        requested = list(model_names) if model_names is not None else self.loaded_model_names
        if not requested:
            if self.load_errors:
                raise RuntimeError(f"No backbones are available: {self.load_errors}")
            raise RuntimeError("No backbones were requested.")

        missing = [name for name in requested if name not in self.models]
        if missing:
            raise ValueError(f"Requested unloaded backbone(s): {missing}")

        array = np.asarray(specs, dtype=np.float32)
        results: dict[str, np.ndarray] = {}

        for model_name in requested:
            model = self.models[model_name]
            model.eval()
            feature_batches: list[np.ndarray] = []

            for start in range(0, len(array), batch_size):
                x = self._prep(array[start : start + batch_size], model_name)
                with torch.no_grad():
                    features = self._pool_output(model(x))
                feature_batches.append(features.detach().cpu().numpy())

            results[model_name] = np.vstack(feature_batches).astype(np.float32)

        return results


def compute_umap_projection(
    features: np.ndarray,
    *,
    scaler: str | None = "standard",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Project a feature matrix into 2D with UMAP after optional scaling."""

    try:
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except ImportError as exc:
        raise ImportError(
            "compute_umap_projection requires scikit-learn. "
            "Install `pip install -e .[anomaly-notebook]`."
        ) from exc

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "compute_umap_projection requires umap-learn. "
            "Install `pip install -e .[anomaly-notebook]`."
        ) from exc

    feature_matrix = np.asarray(features, dtype=np.float32)
    if scaler is None:
        scaled = feature_matrix
    elif scaler == "standard":
        scaled = StandardScaler().fit_transform(feature_matrix)
    elif scaler == "robust":
        scaled = RobustScaler().fit_transform(feature_matrix)
    else:
        raise ValueError("scaler must be one of: None, 'standard', or 'robust'.")

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(scaled).astype(np.float32)
