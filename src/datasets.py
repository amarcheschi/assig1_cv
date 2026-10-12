"""Image sources: synthetic warps + real HPatches pairs (§2.7).

Two complementary protocols (as in the notebook):
(A) synthetic geometric warps of one base image (exact GT homographies);
(B) real HPatches pairs — ``i_*`` illumination vs ``v_*`` viewpoint — with
    shipped ``H_1_k``; plus synthetic photometric warps (H = I) isolating the
    illumination effect.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src import config as C


def load_color(path: str | Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img


def build_synthetic_warps(image: np.ndarray) -> dict:
    """Geometric warps of ``image`` (must match ``homography.build_synthetic_gt``)."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    out = {"Original": image}
    out["Rotated 45°"] = cv2.warpAffine(
        image, cv2.getRotationMatrix2D(center, C.ROT_ANGLE, 1.0), (w, h))
    out["Translated"] = cv2.warpAffine(
        image, np.float32([[1, 0, C.TRANS_TX], [0, 1, C.TRANS_TY]]), (w, h))
    big = cv2.resize(image, None, fx=C.SCALE_UP, fy=C.SCALE_UP,
                      interpolation=cv2.INTER_LINEAR)
    sh, sw = big.shape[:2]
    x0, y0 = (sw - w) // 2, (sh - h) // 2
    out["Scaled 1.5x"] = big[y0:y0 + h, x0:x0 + w]
    small = cv2.resize(image, None, fx=C.SCALE_DOWN, fy=C.SCALE_DOWN,
                        interpolation=cv2.INTER_LINEAR)
    sh2, sw2 = small.shape[:2]
    px, py = (w - sw2) // 2, (h - sh2) // 2
    canvas = np.zeros_like(image)
    canvas[py:py + sh2, px:px + sw2] = small
    out["Scaled 0.5x"] = canvas
    M_c = cv2.getRotationMatrix2D(center, C.COMBINED_ANGLE, C.COMBINED_SCALE)
    M_c[0, 2] += C.COMBINED_TX
    M_c[1, 2] += C.COMBINED_TY
    out["Combined warp"] = cv2.warpAffine(image, M_c, (w, h))
    return out


def photo_warp(bgr: np.ndarray, kind: str) -> np.ndarray:
    """Photometric warp (brightness / contrast / gamma); geometry = identity."""
    f = bgr.astype(np.float32) / 255.0
    if kind == "bright+":
        f = np.clip(f * 1.4 + 0.15, 0, 1)
    elif kind == "dark+":
        f = np.clip(f * 0.5, 0, 1)
    elif kind == "contrast":
        f = np.clip((f - 0.5) * 1.8 + 0.5, 0, 1)
    elif kind == "gamma0.5":
        f = np.clip(f ** 0.5, 0, 1)
    elif kind == "gamma2.0":
        f = np.clip(f ** 2.0, 0, 1)
    else:
        raise ValueError(f"unknown photometric kind: {kind}")
    return (f * 255).astype(np.uint8)


def load_hpatches_pair(seq: str | Path, idx: int):
    """Load real pair ``1.ppm -> idx.ppm`` + shipped ``H_1_idx``.

    Returns ``(img_ref, img_w, H, cat)`` with cat illumination/viewpoint.
    """
    seq = Path(seq)
    img_ref = load_color(seq / "1.ppm")
    img_w = load_color(seq / f"{idx}.ppm")
    H = np.loadtxt(seq / f"H_1_{idx}").astype(float)
    cat = "illumination" if seq.name.startswith("i_") else "viewpoint"
    return img_ref, img_w, H, cat
