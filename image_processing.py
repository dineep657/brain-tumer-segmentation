import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def save_segmentation_overlay_fast(
    raw_image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.45,
    max_side: int = 512,
) -> None:
    """
    Fast CPU overlay for the web app (no Matplotlib).
    Draws a red tumor wash + yellow contour edge onto the MRI and saves PNG.
    """
    gray = np.asarray(raw_image, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("raw_image must be 2D grayscale")

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.shape != gray.shape:
        raise ValueError("mask shape must match raw_image")

    # Normalize to 0-255 uint8
    g_min, g_max = float(gray.min()), float(gray.max())
    if g_max > g_min:
        gray_u8 = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
    else:
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    if mask_u8.any():
        red = np.zeros_like(rgb)
        red[..., 0] = 220.0
        a = float(np.clip(alpha, 0.0, 1.0))
        tumor = mask_u8.astype(bool)
        rgb[tumor] = (1.0 - a) * rgb[tumor] + a * red[tumor]

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    if mask_u8.any():
        # Cheap edge: pixels in mask with a non-mask 4-neighbour
        padded = np.pad(mask_u8, 1, mode="constant")
        edge = (
            (mask_u8 == 1)
            & (
                (padded[1:-1, :-2] == 0)
                | (padded[1:-1, 2:] == 0)
                | (padded[:-2, 1:-1] == 0)
                | (padded[2:, 1:-1] == 0)
            )
        )
        edge_img = Image.fromarray((edge.astype(np.uint8) * 255), mode="L")
        yellow = Image.new("RGB", img.size, (250, 204, 21))
        img.paste(yellow, mask=edge_img)

    # Downscale for web response / free-tier CPU
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)

    img.save(output_path, format="PNG", optimize=True)


def render_segmentation_figure(
    raw_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    show_overlay: bool = True,
    high_contrast: bool = False,
    show_contours: bool = True,
    title: str = "MRI Segmentation Result"
) -> plt.Figure:
    """
    Renders a Matplotlib figure ensuring 100% spatial alignment between the original MRI image,
    the semi-transparent red tumor mask overlay, and the yellow perimeter contour lines.
    
    CRITICAL FIX: Uses explicit 2D meshgrid coordinates (X_grid, Y_grid) for ax.contour()
    to prevent Matplotlib from transposing row/column indices relative to ax.imshow().
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # 1. High Contrast Scaling
    display_image = raw_image.copy()
    if high_contrast:
        p2, p98 = np.percentile(display_image, (2, 98))
        if p98 > p2:
            display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1) * 255.0

    # 2. Render Base Grayscale Image
    ax.imshow(display_image, cmap='gray', origin='upper')
    
    h, w = mask.shape
    
    # 3. Render Semi-Transparent Red Mask Overlay
    if show_overlay and np.any(mask == 1):
        tumor_overlay = np.ma.masked_where(mask != 1, mask)
        ax.imshow(tumor_overlay, cmap='Reds', alpha=alpha, vmin=0, vmax=1, origin='upper')
        
    # 4. Render Perimeter Contour Lines with Explicit Meshgrid Coordinates
    if show_contours and np.any(mask == 1):
        # Create explicit coordinate grid matching (H, W) array indexing
        # X_grid[y, x] = x (horizontal column), Y_grid[y, x] = y (vertical row)
        X_grid, Y_grid = np.meshgrid(np.arange(w), np.arange(h))
        ax.contour(X_grid, Y_grid, mask, levels=[0.5], colors='#FACC15', linewidths=1.8)
        
    ax.set_title(title, color='white', fontsize=12, pad=10, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

def export_overlay_as_png(raw_image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> bytes:
    """
    Exports high-resolution PNG image buffer of the segmented overlay.
    """
    fig = render_segmentation_figure(raw_image, mask, alpha=alpha, show_overlay=True, show_contours=True, title="")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor='#0E1117')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
