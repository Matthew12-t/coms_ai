import io
import logging
import os
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms

from src.models.csrnet import CSRNet

logger = logging.getLogger(__name__)

MAX_IMAGE_DIMENSION = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "csrnet.pth"
HEURISTIC_FACTOR = float(os.getenv("HEURISTIC_FACTOR", "0.0008"))


class _ResizeMaxDimension:
    def __init__(self, max_dim: int) -> None:
        self.max_dim = max_dim

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_dim:
            return image
        scale = self.max_dim / longest
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.BILINEAR)


_transform = transforms.Compose(
    [
        _ResizeMaxDimension(MAX_IMAGE_DIMENSION),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
)


def _resolve_weights_path() -> Path:
    return Path(os.getenv("MODEL_WEIGHTS_PATH") or DEFAULT_WEIGHTS_PATH)


def _maybe_download_weights(target: Path) -> None:
    url = os.getenv("MODEL_WEIGHTS_URL")
    if not url or target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CSRNet weights from %s", url)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
    logger.info("Saved weights to %s", target)


def _load_model() -> tuple[CSRNet | None, bool]:
    target = _resolve_weights_path()
    try:
        _maybe_download_weights(target)
    except Exception as exc:
        logger.warning("Weight download failed: %s", exc)

    if not target.exists():
        logger.warning("CSRNet weights not found at %s — using heuristic fallback", target)
        return None, False

    model = CSRNet()
    state_dict = torch.load(target, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, True


_model, _model_ready = _load_model()


def _heuristic_count(image: Image.Image) -> int:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    if grayscale.size == 0:
        return 0
    gx = np.abs(np.diff(grayscale, axis=1)).mean()
    gy = np.abs(np.diff(grayscale, axis=0)).mean()
    activity = (gx + gy) / 2.0
    estimate = activity * grayscale.size * HEURISTIC_FACTOR
    return max(0, int(estimate))


def predict_crowd(image_bytes: bytes) -> int:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not _model_ready:
        return _heuristic_count(image)
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        density_map = _model(tensor)
    return max(0, int(torch.sum(density_map).item()))


def model_status() -> dict:
    return {
        "model_loaded": _model_ready,
        "weights_path": str(_resolve_weights_path()),
        "fallback": "heuristic" if not _model_ready else None,
    }
