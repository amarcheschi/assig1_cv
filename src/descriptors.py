"""Binary / float descriptors (§2.5).

Covers the OpenCV descriptors used in the notebook plus a small custom
simplified-BRIEF implementation with the two sampling strategies required
by the assignment:

- ``uniform``: point pairs sampled uniformly in the patch (BRIEF paper §3.2, test I).
- ``gaussian``: pairs from an isotropic Gaussian centred on the keypoint
  (BRIEF paper §3.2, test II, sigma = patch_size / 5).

The custom BRIEF is dependency-free (NumPy only) so it always runs, even on
OpenCV builds where ``xfeatures2d`` is missing.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# OpenCV build documentation (§2.5: state explicitly what is available)
# ---------------------------------------------------------------------------
def opencv_build_info() -> dict:
    """Return OpenCV version + which of BRIEF/BRISK/FREAK/SURF is available."""
    info = {"opencv_version": cv2.__version__, "contrib": hasattr(cv2, "xfeatures2d")}
    try:
        cv2.xfeatures2d.BriefDescriptorExtractor_create()
        info["BRIEF"] = "opencv-contrib xfeatures2d"
    except Exception as e:  # noqa: BLE001
        info["BRIEF"] = f"unavailable ({e})"
    try:
        cv2.BRISK_create()
        info["BRISK"] = "available"
    except Exception as e:  # noqa: BLE001
        info["BRISK"] = f"unavailable ({e})"
    try:
        cv2.xfeatures2d.FREAK_create()
        info["FREAK"] = "opencv-contrib xfeatures2d"
    except Exception as e:  # noqa: BLE001
        info["FREAK"] = f"unavailable ({e})"
    try:
        cv2.SIFT_create()
        info["SIFT"] = "available"
    except Exception as e:  # noqa: BLE001
        info["SIFT"] = f"unavailable ({e})"
    info["SURF"] = "skipped (optional per assignment §2.2; needs custom OpenCV build)"
    return info


def descriptor_memory_table(ref_data: dict) -> list:
    """Build dimensionality/memory rows from ``{name: (kp, des)}``.

    Includes the SuperPoint row shape (256×float32) in the docstring for the
    report; rows are computed from actual arrays (incl. ``total_bytes``).
    """
    rows = []
    for name, (kp, des) in sorted(ref_data.items()):
        if des is None:
            rows.append(
                {
                    "Detector": name,
                    "n_features": len(kp),
                    "dim": 0,
                    "dtype": "none",
                    "bytes_per_desc": 0,
                    "total_bytes": 0,
                }
            )
        else:
            rows.append(
                {
                    "Detector": name,
                    "n_features": len(kp),
                    "dim": int(des.shape[1]),
                    "dtype": str(des.dtype),
                    "bytes_per_desc": int(des.itemsize * des.shape[1]),
                    "total_bytes": int(des.nbytes),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Custom simplified BRIEF (Part 3, merged from part3&4.py).
#
# Two sampling strategies per the BRIEF paper §3.2 / assignment §2.5:
# - uniform: pairs uniform in the patch (paper test I)
# - gaussian: isotropic Gaussian pairs centred on the keypoint (paper test II)
# ---------------------------------------------------------------------------
def generate_uniform_pairs(n_bits=256, patch_size=31, seed=0):
    """Uniform sampling pairs, returns ``(pairs1, pairs2)`` each (n_bits, 2).

    ``seed`` makes the pattern reproducible (part3&4.py used unseeded
    ``np.random``; same distribution, deterministic here).
    """
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    p1 = rng.integers(-half, half + 1, (n_bits, 2))
    p2 = rng.integers(-half, half + 1, (n_bits, 2))
    return p1, p2


def generate_gaussian_pairs(n_bits=256, patch_size=31, sigma=5, seed=0):
    """Gaussian sampling pairs (``sigma`` in pixels), same return convention."""
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    p1 = np.clip(rng.normal(0, sigma, (n_bits, 2)), -half, half).astype(int)
    p2 = np.clip(rng.normal(0, sigma, (n_bits, 2)), -half, half).astype(int)
    return p1, p2


def compute_brief(gray, keypoints, pairs1, pairs2):
    """Compare intensities on ``pairs1``/``pairs2`` around each keypoint.

    Returns ``(valid_kps, bits_uint8)`` with one 0/1 row per keypoint; a
    keypoint is dropped when any sample falls outside the image
    (part3&4.py semantics).
    """
    descriptors, valid_kps = [], []
    h, w = gray.shape
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        bits, valid = [], True
        for (dx1, dy1), (dx2, dy2) in zip(pairs1, pairs2):
            x1, y1, x2, y2 = x + dx1, y + dy1, x + dx2, y + dy2
            if x1 < 0 or x1 >= w or x2 < 0 or x2 >= w or y1 < 0 or y1 >= h or y2 < 0 or y2 >= h:
                valid = False
                break
            bits.append(1 if gray[y1, x1] < gray[y2, x2] else 0)
        if valid:
            descriptors.append(bits)
            valid_kps.append(kp)
    if not descriptors:
        return [], np.zeros((0, len(pairs1)), dtype=np.uint8)
    return valid_kps, np.asarray(descriptors, dtype=np.uint8)


def bits_to_bytes(desc):
    """Pack bit rows into bytes (256 bits -> 32 bytes) for Hamming matching."""
    desc = np.asarray(desc, dtype=np.uint8)
    if desc.size == 0:
        return np.zeros((0, 32), dtype=np.uint8)
    return np.packbits(desc, axis=1)


class CustomBrief:
    """Simplified BRIEF: intensity comparisons on pairs around each keypoint.

    Parameters
    ----------
    strategy: 'uniform' | 'gaussian'
    patch_size: side of the square sampling window (default 31, as OpenCV BRIEF).
    n_bits: descriptor length in bits (256 -> 32 bytes, like the notebook).
    seed: RNG seed so sampling patterns are reproducible.
    blur_ksize: Gaussian smoothing applied before sampling (BRIEF paper §3).
    """

    def __init__(
        self,
        strategy: str = "uniform",
        patch_size: int = 31,
        n_bits: int = 256,
        seed: int = 0,
        blur_ksize: int = 5,
    ):
        if strategy not in ("uniform", "gaussian"):
            raise ValueError("strategy must be 'uniform' or 'gaussian'")
        self.strategy = strategy
        self.patch_size = patch_size
        self.n_bits = n_bits
        self.seed = seed
        self.blur_ksize = blur_ksize
        self.pairs = self._make_pairs()

    def _make_pairs(self):
        if self.strategy == "uniform":
            p1, p2 = generate_uniform_pairs(self.n_bits, self.patch_size, self.seed)
        else:  # gaussian, sigma ~ patch/5 per BRIEF paper
            p1, p2 = generate_gaussian_pairs(
                self.n_bits, self.patch_size, self.patch_size / 5.0, self.seed
            )
        return np.stack([p1, p2], axis=1)  # (n_bits, 2, 2)

    def compute(self, gray: np.ndarray, keypoints):
        """Compute descriptors; returns ``(valid_kp, des_uint8)``.

        Keypoints too close to the border are dropped (no valid patch).
        Smoothing follows BRIEF paper §3; sampling delegates to the
        part3&4.py-ported ``compute_brief``/``bits_to_bytes`` above.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        smooth = (
            cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
            if self.blur_ksize > 1
            else gray
        )
        h, w = smooth.shape
        half = self.patch_size // 2
        pre = [
            kp for kp in keypoints
            if half <= kp.pt[0] < w - half and half <= kp.pt[1] < h - half
        ]
        if not pre:
            return [], None
        valid_kp, bits = compute_brief(smooth, pre, self.pairs[:, 0], self.pairs[:, 1])
        if not valid_kp:
            return [], None
        return valid_kp, bits_to_bytes(bits)
