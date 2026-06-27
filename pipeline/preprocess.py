"""Image preprocessing utilities for the OCR pipeline.

All functions operate on OpenCV-style ndarrays. ``preprocess_image`` always
returns a 3-channel BGR uint8 image so that PaddleOCR accepts it directly.
"""

import cv2
import numpy as np


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the page skew angle (degrees) of a 2D uint8 grayscale image.

    Uses the minimum-area rectangle enclosing the foreground (dark) pixels and
    the OpenCV >= 4.5 angle convention (0, 90]. Clamped to ``[-15, 15]``; returns
    ``0.0`` when there isn't enough foreground to estimate reliably.
    """
    if gray is None or gray.ndim != 2:
        return 0.0
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    # Otsu on the inverted image: dark glyph pixels become nonzero coordinates.
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 10:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV >= 4.5 returns the angle in (0, 90]; map to a small signed
    # deviation from horizontal in (-45, 45].
    if angle > 45:
        angle -= 90.0
    return float(max(-15.0, min(15.0, angle)))


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D grayscale image by ``angle`` degrees, white-filling borders."""
    h, w = gray.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        gray, rot_mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small page skew on a 2D uint8 grayscale image.

    Only small angles are corrected (estimate clamped to ``[-15, 15]``) to avoid
    catastrophic rotations on noisy inputs. Rotation fills exposed borders white.
    """
    if gray is None or gray.ndim != 2:
        raise ValueError("deskew expects a 2D grayscale uint8 array")
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return gray

    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.1:
        return gray
    return _rotate(gray, angle)


def _sauvola_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image with Sauvola's local thresholding.

    Sauvola adapts the threshold per-pixel using the local mean and standard
    deviation, which handles uneven lighting and faint text far better than a
    single global or simple adaptive-Gaussian threshold — the usual choice for
    document OCR. Falls back to adaptive-Gaussian if scikit-image is missing.
    """
    try:
        from skimage.filters import threshold_sauvola

        # window_size must be odd; 25 suits body text at ~300 DPI.
        thresh = threshold_sauvola(gray, window_size=25, k=0.2)
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary
    except Exception:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=15,
        )


def preprocess_image(
    img_bgr: np.ndarray, *, mode: str, binarize: bool
) -> np.ndarray:
    """Preprocess a BGR image for OCR and return a 3-channel BGR uint8 image.

    Steps (in order):
      1. Convert to grayscale.
      2. Deskew (small-angle correction with white border).
      3. CLAHE contrast equalization (rescues faint / unevenly-lit scans).
      4. Denoise via ``cv2.fastNlMeansDenoising`` ONLY when ``mode == "max"``.
      5. Sauvola binarization ONLY when ``binarize`` is True.
      6. Pad a white border so text touching the page edge is still detected.

    The result is always converted back to 3-channel BGR so PaddleOCR accepts
    it; this function never returns a 2D array.

    Parameters
    ----------
    img_bgr:
        HxWx3 BGR uint8 image.
    mode:
        ``"max"`` or ``"fast"``.
    binarize:
        Whether to apply Sauvola binarization.
    """
    if img_bgr is None:
        raise ValueError("preprocess_image received None")

    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    # 1. Grayscale.
    if img_bgr.ndim == 2:
        gray = img_bgr
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("preprocess_image expects a BGR(A) or grayscale image")

    # 2. Deskew — estimate the angle once so we can both correct it and decide
    #    whether sharpening is safe below.
    skew = estimate_skew_angle(gray)
    if abs(skew) >= 0.1:
        gray = _rotate(gray, skew)

    # 3. CLAHE — local contrast equalization. Mild clip so clean renders are
    #    barely touched while faint/grey scans get a real lift.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Denoise (only in max mode — it is the slow step).
    if mode == "max":
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 4b. Unsharp-mask sharpening to crisp thin strokes — improves glyph
    #     separation (e.g. capital "I" vs lowercase "l") on soft scans, which
    #     measurably helped a real ID-card scan. ONLY when the page is near-flat:
    #     on a heavily-skewed page, deskew's rotation leaves interpolation blur
    #     that sharpening would amplify, hurting recognition. Benchmarked: gated
    #     this way it helps flat scans with no regression on rotated pages.
    if abs(skew) < 4.0:
        _blur = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, _blur, -0.5, 0)

    # 5. Sauvola binarize (only when requested).
    if binarize:
        gray = _sauvola_binarize(gray)

    # 6. White border padding so edge-touching text isn't clipped by detection.
    gray = cv2.copyMakeBorder(
        gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255
    )

    # Always return 3-channel BGR uint8.
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr
