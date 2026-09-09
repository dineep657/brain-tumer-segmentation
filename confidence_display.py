"""
confidence_display.py — Confidence Threshold Module

Encapsulates the confidence threshold constant and the label-resolution logic
for the Brain Tumor Segmentation & Classification app. By centralising these
here, app.py has no inline threshold comparisons and the threshold value is
defined exactly once.

Public API:
    CONFIDENCE_THRESHOLD          — float constant (0.60)
    LOW_CONFIDENCE_WARNING_TEXT   — warning label string
    resolve_tumor_type_label()    — pure function mapping scan state → display label
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.60
LOW_CONFIDENCE_WARNING_TEXT: str = "⚠️ LOW CONFIDENCE — Refer to specialist"


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

def resolve_tumor_type_label(
    tumor_type: str,
    confidence: float | None,
    is_valid_mri: bool,
    tumor_detected: bool,
) -> tuple[str, bool]:
    """Return ``(display_label, is_low_confidence)`` for the tumor-type card.

    Priority rules (evaluated in order):

    1. ``is_valid_mri`` is ``False``  → ``("Invalid Input", False)``
    2. ``tumor_detected`` is ``False`` → ``("No Tumor", False)``
    3. ``confidence`` is not ``None`` and ``confidence < CONFIDENCE_THRESHOLD``
       (after clamping to ``[0, 1]``)                → ``(LOW_CONFIDENCE_WARNING_TEXT, True)``
    4. Otherwise                                      → ``(tumor_type, False)``

    Args:
        tumor_type:      The classifier's top predicted class name
                         (e.g. "Glioma", "Meningioma", "Pituitary").
        confidence:      Softmax probability for the top class in ``[0, 1]``,
                         or ``None`` when the classifier was not executed
                         (e.g. no-tumor path).
        is_valid_mri:    ``True`` when the uploaded image passed the MRI
                         validity check.
        tumor_detected:  ``True`` when the UNet segmentation detected a tumor.

    Returns:
        A 2-tuple ``(display_label, is_low_confidence)`` where
        ``is_low_confidence`` is ``True`` only when the low-confidence warning
        is shown.
    """
    # Rule 1 — invalid input takes highest priority
    if not is_valid_mri:
        return ("Invalid Input", False)

    # Rule 2 — no tumor detected; confidence is irrelevant
    if not tumor_detected:
        return ("No Tumor", False)

    # Rule 3 — low confidence warning (only when confidence is provided)
    if confidence is not None:
        # Clamp out-of-range values to [0, 1] before comparison
        clamped = confidence
        if confidence < 0.0 or confidence > 1.0:
            logger.warning(
                "confidence value %r is outside [0, 1]; clamping to [0, 1].",
                confidence,
            )
            clamped = max(0.0, min(1.0, confidence))

        if clamped < CONFIDENCE_THRESHOLD:
            return (LOW_CONFIDENCE_WARNING_TEXT, True)

    # Rule 4 — confident positive result
    return (tumor_type, False)
