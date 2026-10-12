"""Keypoint detectors + detect/describe helpers (§2.2–§2.4).

Wraps the OpenCV detectors used in the notebook so every experiment shares
one factory. FAST has no descriptor of its own: the ``FAST+*`` entries pair
a FAST detector (applied at match time) with a binary descriptor object.
"""

from __future__ import annotations

import cv2

from src import config as C
from src.descriptors import CustomBrief


def build_detectors(orb_nfeatures: int = 5000):
    """Return ``{name: (det, norm, desc_obj)}``.

    - Classic entries (SIFT/ORB/KAZE): ``det`` is a full Feature2D,
      ``desc_obj`` is None → use ``det.detectAndCompute``.
    - ``FAST+*`` entries: ``det`` is None (FAST created per call so the
      caller can set NMS), ``desc_obj`` computes on FAST keypoints.
    - ``norm`` is the matching norm: L2 for float, Hamming for binary (§2.5).
    """
    return {
        "SIFT": (cv2.SIFT_create(), cv2.NORM_L2, None),
        "ORB": (cv2.ORB_create(nfeatures=orb_nfeatures), cv2.NORM_HAMMING, None),
        "KAZE": (cv2.KAZE_create(), cv2.NORM_L2, None),
        "FAST+BRIEF": (
            None,
            cv2.NORM_HAMMING,
            cv2.xfeatures2d.BriefDescriptorExtractor_create(),
        ),
        "FAST+BRISK": (None, cv2.NORM_HAMMING, cv2.BRISK_create()),
        "FAST+FREAK": (
            None,
            cv2.NORM_HAMMING,
            cv2.xfeatures2d.FREAK_create(),
        ),
        # Custom simplified BRIEF, both sampling strategies (Part 3, §2.5).
        # CustomBrief.compute(gray, kp) follows the desc_obj protocol.
        "FAST+cBRIEF-uniform": (
            None,
            cv2.NORM_HAMMING,
            CustomBrief(strategy="uniform"),
        ),
        "FAST+cBRIEF-gaussian": (
            None,
            cv2.NORM_HAMMING,
            CustomBrief(strategy="gaussian"),
        ),
    }


def detect_and_describe(det, desc_obj, gray):
    """Detect + describe a grayscale image.

    Returns ``(keypoints, descriptors)`` where descriptors may be None.
    FAST keypoints get a fixed size/octave and no orientation so that
    BRISK/FREAK behave deterministically (as in the notebook).
    """
    if desc_obj is None:
        try:
            return det.detectAndCompute(gray, None)
        except cv2.error:
            return [], None
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(True)
    kp = fast.detect(gray, None)
    if not kp:
        return [], None
    for k in kp:
        k.size = C.FAST_PATCH_SIZE
        k.octave = 0
        k.angle = -1
    try:
        return desc_obj.compute(gray, kp)
    except cv2.error:
        return kp, None


def detect_only(det, desc_obj, gray):
    """Detector-only keypoints (for repeatability, no descriptor)."""
    kp, _ = detect_and_describe(det, desc_obj, gray)
    # detect_and_describe already ran description; cheap enough, but for
    # pure detectors avoid the describe cost via direct detect.
    if desc_obj is None:
        return kp
    return kp


def detector_keypoint_counts(gray, detectors=None):
    """Return ``{name: (annotated_bgr, n)}`` for a quick detection overview."""
    import cv2 as _cv  # local import keeps module import light

    detectors = detectors or build_detectors()
    out = {}
    for name, (det, _norm, desc_obj) in detectors.items():
        kp, _ = detect_and_describe(det, desc_obj, gray)
        vis = _cv.drawKeypoints(
            _cv.cvtColor(gray, _cv.COLOR_GRAY2BGR),
            kp,
            None,
            color=(0, 255, 0),
            flags=_cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        out[name] = (vis, len(kp))
    return out
