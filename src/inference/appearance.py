from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class AppearanceEmbedder(Protocol):
    def embed(self, frame, detections: list[object]) -> dict[int, np.ndarray]: ...


@dataclass(slots=True)
class HistogramAppearanceEmbedder:
    bins: tuple[int, int] = (8, 8)

    def embed(self, frame, detections: list[object]) -> dict[int, np.ndarray]:
        if frame is None:
            return {}

        embeddings: dict[int, np.ndarray] = {}
        for det_idx, detection in enumerate(detections):
            crop = _extract_crop(frame, detection.bbox)
            if crop is None:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = (
                cv2.calcHist([hsv], [0, 1], None, list(self.bins), [0, 180, 0, 256])
                .flatten()
                .astype(np.float32)
            )
            embeddings[det_idx] = _normalize_embedding(hist)
        return embeddings


class TorchvisionReIDEmbedder:
    def __init__(
        self,
        *,
        model_name: str = "mobilenet_v3_small",
        device: str = "cpu",
        pretrained: bool = True,
        weights_path: str = "",
        input_size: int = 128,
    ) -> None:
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError(
                "torch and torchvision are required for learned appearance embeddings."
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        self._model = self._build_model(
            models=models,
            model_name=model_name,
            pretrained=pretrained,
            weights_path=weights_path,
        )
        self._model.to(self._device)
        self._model.eval()
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def embed(self, frame, detections: list[object]) -> dict[int, np.ndarray]:
        if frame is None:
            return {}

        embeddings: dict[int, np.ndarray] = {}
        with self._torch.inference_mode():
            for det_idx, detection in enumerate(detections):
                crop = _extract_crop(frame, detection.bbox)
                if crop is None:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                vector = self._model(tensor).flatten().detach().cpu().numpy()
                embeddings[det_idx] = _normalize_embedding(vector)
        return embeddings

    def _build_model(
        self, *, models, model_name: str, pretrained: bool, weights_path: str
    ):
        if model_name != "mobilenet_v3_small":
            raise ValueError(f"Unsupported appearance model '{model_name}'.")

        weights = None
        if weights_path:
            model = models.mobilenet_v3_small(weights=None)
            state_dict = self._torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            if pretrained:
                try:
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    model = models.mobilenet_v3_small(weights=weights)
                except Exception as exc:
                    LOGGER.warning(
                        "Falling back from pretrained MobileNetV3 appearance encoder: %s",
                        exc,
                    )
                    model = models.mobilenet_v3_small(weights=None)
            else:
                model = models.mobilenet_v3_small(weights=None)

        model.classifier = self._torch.nn.Identity()
        return model


def build_appearance_embedder(config: dict) -> AppearanceEmbedder:
    appearance_model = str(config.get("appearance_model", "mobilenet_v3_small")).lower()
    if appearance_model == "histogram":
        return HistogramAppearanceEmbedder()

    if appearance_model == "mobilenet_v3_small":
        try:
            return TorchvisionReIDEmbedder(
                model_name="mobilenet_v3_small",
                device=str(config.get("appearance_device", "cpu")),
                pretrained=bool(config.get("appearance_pretrained", True)),
                weights_path=str(config.get("appearance_weights_path", "")),
                input_size=int(config.get("appearance_input_size", 128)),
            )
        except Exception as exc:
            LOGGER.warning(
                "Falling back to histogram appearance embeddings: %s",
                exc,
            )
            return HistogramAppearanceEmbedder()

    raise ValueError(f"Unsupported appearance_model '{appearance_model}'.")


def _extract_crop(frame, bbox: tuple[float, float, float, float]):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(x1)))
    iy1 = max(0, min(height - 1, int(y1)))
    ix2 = max(ix1 + 1, min(width, int(x2)))
    iy2 = max(iy1 + 1, min(height, int(y2)))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    return crop


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    embedding = vector.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding
