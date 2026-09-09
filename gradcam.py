"""
gradcam.py — Grad-CAM Computation Module

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for the
LightweightClassifier2D model. Computation is isolated from the production
inference path: the model's eval() state and weights are never mutated.

Requirements: 3.1, 3.3, 3.6, 3.7, 3.8, 3.9
"""

import logging
import warnings

import numpy as np
import torch
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 6.1 — generate_gradcam
# ---------------------------------------------------------------------------

def generate_gradcam(model, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Compute a Grad-CAM activation map for the given class index.

    Parameters
    ----------
    model : LightweightClassifier2D
        The classifier in eval() mode.
    tensor : torch.Tensor
        Shape (1, 1, 128, 128) — pre-processed classifier input, on the same
        device as the model.
    class_idx : int
        Index of the target class (0-based).

    Returns
    -------
    np.ndarray
        2-D float32 array of shape (H_feat, W_feat) with values in [0, 1].
        For a 128×128 input the feature map is 16×16 (after prep MaxPool +
        two stride-2 downs).

    Notes
    -----
    - Both hooks are removed in a `finally` block regardless of errors.
    - The model remains in eval() mode throughout; no parameter update occurs.
    - If gradients are None (e.g. no grad flow), a zero-filled map is returned
      and a warning is logged (acceptable degraded behaviour per design.md).

    Validates: Requirements 3.1, 3.7
    """
    feature_maps: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fwd_handle = model.res3.register_forward_hook(forward_hook)
    bwd_handle = model.res3.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            # Ensure the tensor participates in the computation graph
            inp = tensor.requires_grad_(True) if not tensor.requires_grad else tensor
            logits = model(inp)                    # (1, num_classes)
            score = logits[0, class_idx]           # scalar
            model.zero_grad()
            score.backward()

        if not gradients or gradients[0] is None:
            logger.warning(
                "generate_gradcam: gradients were None for class_idx=%d. "
                "Returning zero-filled activation map.",
                class_idx,
            )
            fmaps = feature_maps[0].detach()
            H, W = fmaps.shape[2], fmaps.shape[3]
            return np.zeros((H, W), dtype=np.float32)

        # ---- Grad-CAM computation ----
        # feature_maps: (1, C, H, W)   gradients: (1, C, H, W)
        fmaps = feature_maps[0].detach()          # (1, C, H, W)
        grads = gradients[0].detach()             # (1, C, H, W)

        # Channel-wise weights = global average pool of gradients over spatial dims
        # shape: (1, C, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum over channels → (1, 1, H, W)
        weighted_sum = (weights * fmaps).sum(dim=1, keepdim=True)

        # ReLU — keep only positive influence
        cam = torch.relu(weighted_sum).squeeze()  # (H, W)

        cam_np = cam.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            # Flat map — return as-is (all zeros or a uniform constant)
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# ---------------------------------------------------------------------------
# Task 6.2 — resize_activation_map
# ---------------------------------------------------------------------------

def resize_activation_map(
    activation: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Resize a 2-D activation map to (target_h, target_w) using bilinear
    interpolation via PIL.Image.resize.

    Parameters
    ----------
    activation : np.ndarray
        2-D float array, any shape.
    target_h : int
        Target height in pixels.
    target_w : int
        Target width in pixels.

    Returns
    -------
    np.ndarray
        float32 array of shape (target_h, target_w) with values in [0, 1].

    Validates: Requirements 3.6
    """
    # PIL expects (width, height) for the size argument
    pil_img = Image.fromarray(activation.astype(np.float32))
    pil_resized = pil_img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    resized = np.array(pil_resized, dtype=np.float32)

    # Re-normalise after resize to guarantee [0, 1] output
    r_min = resized.min()
    r_max = resized.max()
    if r_max - r_min > 1e-8:
        resized = (resized - r_min) / (r_max - r_min)
    else:
        resized = np.zeros_like(resized)

    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Task 6.2 — blend_heatmap_with_image
# ---------------------------------------------------------------------------

def blend_heatmap_with_image(
    grayscale_image: np.ndarray,
    activation_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a Grad-CAM activation map with the original grayscale MRI image.

    Parameters
    ----------
    grayscale_image : np.ndarray
        Shape (H, W), float values in 0–255.
    activation_map : np.ndarray
        Shape (H, W), float values in [0, 1].
    alpha : float
        Blend weight for the heatmap layer (default 0.5).
    colormap : str
        Matplotlib colormap name (default "jet").

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint8, values in [0, 255].

    Validates: Requirements 3.3
    """
    # 1. Apply matplotlib colormap to activation_map → (H, W, 4) RGBA in [0, 1]
    cmap = plt.get_cmap(colormap)
    rgba = cmap(activation_map)                    # (H, W, 4) float64 [0, 1]
    rgb_heatmap = rgba[:, :, :3] * 255.0           # (H, W, 3) float64 [0, 255]

    # 2. Convert grayscale to 3-channel RGB
    gray_clipped = np.clip(grayscale_image, 0.0, 255.0)
    rgb_image = np.stack([gray_clipped] * 3, axis=-1)  # (H, W, 3)

    # 3. Blend
    blended = (1.0 - alpha) * rgb_image + alpha * rgb_heatmap

    # 4. Clip and cast
    result = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Task 6.3 — get_gradcam_panel_title
# ---------------------------------------------------------------------------

def get_gradcam_panel_title(predicted_class: str, is_low_confidence: bool) -> str:
    """
    Build the Grad-CAM Visualization_Panel title string.

    Parameters
    ----------
    predicted_class : str
        The top predicted class name (e.g. "Glioma").
    is_low_confidence : bool
        When True, appends " (Low Confidence)" to the title.

    Returns
    -------
    str
        "🔥 Grad-CAM Explainability — {predicted_class}"
        or
        "🔥 Grad-CAM Explainability — {predicted_class} (Low Confidence)"

    Validates: Requirements 3.8, 3.9
    """
    title = f"🔥 Grad-CAM Explainability \u2014 {predicted_class}"
    if is_low_confidence:
        title += " (Low Confidence)"
    return title
