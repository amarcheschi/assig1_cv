"""Descriptor matching: ratio test, one-to-one, learned matchers (§2.6, §3.1)."""

from __future__ import annotations

import cv2
import numpy as np

from src import config as C
from src.homography import project_points


def ratio_match(des1, des2, norm, ratio: float = C.RATIO):
    """kNN (k=2) + Lowe ratio test. Returns ``(good, n_raw)``."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    return good, len(raw)


def unique_matches(matches, query_key="queryIdx", train_key="trainIdx"):
    """Enforce one-to-one pairing (greedy by ascending distance).

    The ratio test alone does not guarantee uniqueness, which the PMR /
    precision protocol (§2.8) requires.
    """
    matches = sorted(matches, key=lambda m: m.distance)
    seen_q, seen_t, kept = set(), set(), []
    for m in matches:
        q, t = getattr(m, query_key), getattr(m, train_key)
        if q in seen_q or t in seen_t:
            continue
        seen_q.add(q)
        seen_t.add(t)
        kept.append(m)
    return kept


def crosscheck_match(des1, des2, norm):
    """Mutual-nearest-neighbour (cross-check) matching, then one-to-one.

    Used by the part3&4.py experiments for both custom BRIEF (Hamming) and
    SuperPoint+NN (L2). Returns the one-to-one ``good`` list.
    """
    if des1 is None or des2 is None or len(des1) < 1 or len(des2) < 1:
        return []
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return []
    return unique_matches(sorted(matches, key=lambda m: m.distance))


def xy_to_keypoints(xy, size=8.0):
    """Convert an (N,2) array of (x, y) points to a ``cv2.KeyPoint`` list."""
    return [cv2.KeyPoint(float(x), float(y), float(size)) for x, y in np.asarray(xy)]


def match_pair(kp_ref, des_ref, kp_w, des_w, norm, ratio: float = C.RATIO):
    """Match one pair end-to-end: ratio test + one-to-one. Returns ``good_u``."""
    good, _ = ratio_match(des_ref, des_w, norm, ratio)
    if not good:
        return []
    return unique_matches(good)


def split_correct(kp_ref, kp_w, H, good_u, thresh: float = C.CORRECT_T):
    """Split putative matches into (correct, incorrect) at ``thresh`` px."""
    ref_xy = np.array([k.pt for k in kp_ref], dtype=np.float64)
    warp_xy = np.array([k.pt for k in kp_w], dtype=np.float64)
    proj = project_points(H, ref_xy)
    ok, bad = [], []
    for m in good_u:
        e = float(np.linalg.norm(proj[m.queryIdx] - warp_xy[m.trainIdx]))
        (ok if e <= thresh else bad).append(m)
    return ok, bad


# ---------------------------------------------------------------------------
# Learned matchers (§2.6). Classical path above is the default; these are
# thin adapters so SuperPoint/SuperGlue/LightGlue plug into the same
# evaluate_pair / homography code. They import lazily and raise a clear
# error when the optional dependency is missing.
# ---------------------------------------------------------------------------
def superpoint_nnnp(kp0_xy, des0, kp1_xy, des1):
    """Mutual-nearest-neighbour matching for SuperPoint float descriptors."""
    if len(des0) == 0 or len(des1) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des0.astype(np.float32), des1.astype(np.float32))
    return unique_matches(sorted(matches, key=lambda m: m.distance))


def run_superglue_pair(img0_bgr, img1_bgr, weights="outdoor",
                       match_threshold=0.2, sinkhorn_iterations=20,
                       max_keypoints=1024):
    """SuperPoint + SuperGlue on an image pair (Magicleap pretrained repo).

    Returns ``(kp0_cv, kp1_cv, good_u)`` with cv2.KeyPoint / DMatch objects so
    downstream ``evaluate_pair`` / ``homography_metrics`` work unchanged.
    Checkpoint: SuperGluePretrainedNetwork ``{weights}`` (indoor/outdoor).
    """
    import torch  # lazy: learned path is optional

    from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
    from SuperGluePretrainedNetwork.models.superglue import SuperGlue

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = SuperPoint({"nms_radius": 4, "keypoint_threshold": 0.005,
                     "max_keypoints": max_keypoints}).eval().to(device)
    sg = SuperGlue({"weights": weights, "sinkhorn_iterations": sinkhorn_iterations,
                    "match_threshold": match_threshold}).eval().to(device)

    def _prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return torch.from_numpy(g)[None, None].to(device)

    with torch.no_grad():
        p0, p1 = sp({"image": _prep(img0_bgr)}), sp({"image": _prep(img1_bgr)})
        data = {"image0": _prep(img0_bgr), "image1": _prep(img1_bgr),
                **{k + "0": v for k, v in p0.items()},
                **{k + "1": v for k, v in p1.items()}}
        pred = {k: v[0].cpu().numpy() for k, v in sg(data).items()
                if isinstance(v, torch.Tensor)}

    kpts0, kpts1 = p0["keypoints"][0].cpu().numpy(), p1["keypoints"][0].cpu().numpy()
    matches, conf = pred["matches0"], pred["matching_scores0"]
    kp0_cv = [cv2.KeyPoint(float(x), float(y), 8) for x, y in kpts0]
    kp1_cv = [cv2.KeyPoint(float(x), float(y), 8) for x, y in kpts1]
    good = [cv2.DMatch(i, int(m), float(1.0 - c))
            for i, (m, c) in enumerate(zip(matches, conf)) if m > -1]
    return kp0_cv, kp1_cv, unique_matches(good)


# ---------------------------------------------------------------------------
# HuggingFace learned path (merged from part3&4.py):
# SuperPoint (magic-leap-community/superpoint) + LightGlue
# (ETH-CVG/lightglue_superpoint) via transformers. SuperGlue above stays
# available through the Magicleap repo; this file's experiments use SP+NN
# and LightGlue, hence "minus superglue".
# ---------------------------------------------------------------------------
_HF_CACHE: dict = {}


def _hf_device():
    import torch  # lazy: learned path is optional

    return "cuda" if torch.cuda.is_available() else "cpu"


def superpoint_features_hf(img_bgr, model_id="magic-leap-community/superpoint",
                           device=None):
    """SuperPoint keypoints + descriptors via HF transformers.

    Mirrors part3&4.py ``superpoint_features``. Returns ``(kps_xy, descs)``
    as float32 numpy arrays, ``kps_xy`` shaped (N,2) in original pixels.
    Models are cached process-wide after the first download/load.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SuperPointForKeypointDetection

    device = device or _hf_device()
    key = ("sp", model_id, device)
    if key not in _HF_CACHE:
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = SuperPointForKeypointDetection.from_pretrained(model_id).to(device)
        model.eval()
        _HF_CACHE[key] = (proc, model)
    proc, model = _HF_CACHE[key]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = proc(pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    result = proc.post_process_keypoint_detection(out, [(pil.height, pil.width)])[0]
    return (result["keypoints"].cpu().numpy().astype(np.float32),
            result["descriptors"].cpu().numpy().astype(np.float32))


def lightglue_match_hf(img0_bgr, img1_bgr,
                       model_id="ETH-CVG/lightglue_superpoint",
                       threshold=0.1, device=None):
    """LightGlue correspondences via HF transformers (part3&4.py logic).

    Returns ``(pts0, pts1)`` float32 arrays, row i ↔ row i, in original
    pixels. LightGlue is the *matcher* here: same SuperPoint features as
    SP+NN, different correspondence stage (§2.6).
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = device or _hf_device()
    key = ("lg", model_id, device)
    if key not in _HF_CACHE:
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        _HF_CACHE[key] = (proc, model)
    proc, model = _HF_CACHE[key]
    a = Image.fromarray(cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB))
    b = Image.fromarray(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc([[a, b]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    result = proc.post_process_keypoint_matching(
        out, [[(a.height, a.width), (b.height, b.width)]],
        threshold=threshold)[0]
    return (result["keypoints0"].cpu().numpy().astype(np.float32),
            result["keypoints1"].cpu().numpy().astype(np.float32))


def run_lightglue_pair(img0_bgr, img1_bgr, threshold=0.1, device=None,
                       model_id="ETH-CVG/lightglue_superpoint",
                       sp_model_id="magic-leap-community/superpoint"):
    """SuperPoint + LightGlue on an image pair (HF transformers backend).

    Returns ``(kp0_cv, kp1_cv, good_u)`` with ``cv2.KeyPoint`` / ``DMatch``
    objects (``DMatch(i, i, 0.0)``: LightGlue outputs already-paired point
    lists) so downstream ``evaluate_pair`` / ``homography_metrics`` work
    unchanged. ``kp*_cv`` are the *matched* points; the PMR denominator
    (features detected) comes from the standalone SuperPoint extraction —
    see ``evaluation.evaluate_learned_subset``.
    """
    pts0, pts1 = lightglue_match_hf(img0_bgr, img1_bgr, model_id, threshold, device)
    kp0_cv, kp1_cv = xy_to_keypoints(pts0), xy_to_keypoints(pts1)
    good = [cv2.DMatch(i, i, 0.0) for i in range(len(pts0))]
    return kp0_cv, kp1_cv, unique_matches(good)
