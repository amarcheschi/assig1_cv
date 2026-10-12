"""Quantitative evaluation: PMR/precision/score/recall, repeatability (§2.7–§2.8).

Same protocol everywhere: RATIO=0.8, CORRECT_T=3.0px, one-to-one matching.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import config as C
from src.datasets import build_synthetic_warps, load_color, load_hpatches_pair, photo_warp
from src.detectors import build_detectors, detect_and_describe
from src.descriptors import descriptor_memory_table
from src.homography import build_synthetic_gt, homography_metrics, project_points
from src.matching import match_pair, xy_to_keypoints


# ---------------------------------------------------------------------------
# Core per-pair metrics
# ---------------------------------------------------------------------------
def evaluate_pair(kp_ref, kp_warp, H, img_warp_shape, matches, correct_thresh=C.CORRECT_T):
    """PMR / Precision / MatchingScore / Recall for one pair (§2.8).

    ``matches`` must already be one-to-one. ``H`` maps ref -> warp.
    """
    n_ref = len(kp_ref)
    if n_ref == 0 or len(kp_warp) == 0:
        return dict(n_features=n_ref, n_putative=0, n_correct=0,
                    n_correspondences=0, PMR=0.0, Precision=0.0,
                    MatchingScore=0.0, Recall=0.0)
    ref_xy = np.array([k.pt for k in kp_ref], dtype=np.float64)
    warp_xy = np.array([k.pt for k in kp_warp], dtype=np.float64)
    proj_ref = project_points(H, ref_xy)
    hw, ww = img_warp_shape[:2]
    inside = ((proj_ref[:, 0] >= 0) & (proj_ref[:, 0] < ww)
              & (proj_ref[:, 1] >= 0) & (proj_ref[:, 1] < hw))
    n_corr = int(inside.sum())
    n_put = len(matches)
    n_ok = sum(1 for m in matches
               if np.linalg.norm(proj_ref[m.queryIdx] - warp_xy[m.trainIdx]) <= correct_thresh)
    return dict(
        n_features=n_ref, n_putative=n_put, n_correct=n_ok, n_correspondences=n_corr,
        PMR=n_put / n_ref if n_ref else 0.0,
        Precision=n_ok / n_put if n_put else 0.0,
        MatchingScore=n_ok / n_ref if n_ref else 0.0,
        Recall=n_ok / n_corr if n_corr else 0.0,
    )


def detector_repeatability(kp_ref, kp_w, H, warp_shape, eps=C.CORRECT_T):
    """Detector-only repeatability: projected ref kp with a target kp < eps."""
    n_ref = len(kp_ref)
    if n_ref == 0 or len(kp_w) == 0:
        return dict(n_ref=n_ref, n_valid=0, n_repeatable=0, repeatability=0.0)
    ref_xy = np.array([k.pt for k in kp_ref], dtype=np.float64)
    warp_xy = np.array([k.pt for k in kp_w], dtype=np.float64)
    proj = project_points(H, ref_xy)
    h2, w2 = warp_shape[:2]
    inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w2)
              & (proj[:, 1] >= 0) & (proj[:, 1] < h2))
    n_valid = int(inside.sum())
    if n_valid == 0:
        return dict(n_ref=n_ref, n_valid=0, n_repeatable=0, repeatability=0.0)
    dists, _ = cKDTree(warp_xy).query(proj[inside], k=1)
    n_rep = int((dists <= eps).sum())
    return dict(n_ref=n_ref, n_valid=n_valid, n_repeatable=n_rep,
                repeatability=n_rep / n_valid if n_valid else 0.0)


# ---------------------------------------------------------------------------
# Full experiment drivers (return DataFrames; main.py saves them)
# ---------------------------------------------------------------------------
def _match_images(gray_ref, gray_w, det, norm, desc_obj, ratio=C.RATIO):
    kp_ref, des_ref = detect_and_describe(det, desc_obj, gray_ref)
    kp_w, des_w = detect_and_describe(det, desc_obj, gray_w)
    good_u = match_pair(kp_ref, des_ref, kp_w, des_w, norm, ratio)
    return kp_ref, kp_w, good_u


def evaluate_synthetic(base_image, detectors=None, ratio=C.RATIO,
                       correct_t=C.CORRECT_T, warps=None) -> pd.DataFrame:
    """Synthetic viewpoint warps of one base image (exact GT)."""
    detectors = detectors or build_detectors()
    images = build_synthetic_warps(base_image)
    if warps is not None:
        images = {k: v for k, v in images.items() if k in ("Original", *warps)}
    gt = build_synthetic_gt(base_image.shape)
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        gray_ref = cv2.cvtColor(images["Original"], cv2.COLOR_BGR2GRAY)
        for warp_name, img_w in images.items():
            if warp_name == "Original":
                continue
            kp_ref, kp_w, good_u = _match_images(
                gray_ref, cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY),
                det, norm, desc_obj, ratio)
            rows.append({"Detector": det_name, "Warp": warp_name,
                         **evaluate_pair(kp_ref, kp_w, gt[warp_name],
                                         img_w.shape, good_u, correct_t)})
    cols = ["Detector", "Warp", "n_features", "n_putative", "n_correct",
            "n_correspondences", "PMR", "Precision", "MatchingScore", "Recall"]
    return pd.DataFrame(rows)[cols].sort_values(["Detector", "Warp"]).reset_index(drop=True)


def evaluate_repeatability_synthetic(base_image, detectors=None,
                                     eps=C.CORRECT_T, warps=None) -> pd.DataFrame:
    detectors = detectors or build_detectors()
    images = build_synthetic_warps(base_image)
    if warps is not None:
        images = {k: v for k, v in images.items() if k in ("Original", *warps)}
    gt = build_synthetic_gt(base_image.shape)
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        gray_ref = cv2.cvtColor(images["Original"], cv2.COLOR_BGR2GRAY)
        kp_ref, _ = detect_and_describe(det, desc_obj, gray_ref)
        for warp_name, img_w in images.items():
            if warp_name == "Original":
                continue
            kp_w, _ = detect_and_describe(
                det, desc_obj, cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY))
            rows.append({"Detector": det_name, "Warp": warp_name,
                         **detector_repeatability(kp_ref, kp_w, gt[warp_name],
                                                  img_w.shape, eps)})
    return pd.DataFrame(rows).sort_values(["Detector", "Warp"]).reset_index(drop=True)


def evaluate_homography_synthetic(base_image, detectors=None,
                                   ransac_t=C.RANSAC_T) -> pd.DataFrame:
    """RANSAC homography quality per detector x warp (§3.2)."""
    detectors = detectors or build_detectors()
    images = build_synthetic_warps(base_image)
    gt = build_synthetic_gt(base_image.shape)
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        gray_ref = cv2.cvtColor(images["Original"], cv2.COLOR_BGR2GRAY)
        for warp_name, img_w in images.items():
            if warp_name == "Original":
                continue
            kp_ref, kp_w, good_u = _match_images(
                gray_ref, cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), det, norm, desc_obj)
            m = homography_metrics(kp_ref, kp_w, good_u, gt[warp_name],
                                   ransac_t, img_w.shape)
            m.pop("H_est", None)
            rows.append({"Detector": det_name, "Warp": warp_name, **m})
    return pd.DataFrame(rows).sort_values(["Detector", "Warp"]).reset_index(drop=True)


def evaluate_real_hpatches(detectors=None, seqs=C.REAL_SEQS, idxs=C.REAL_IDXS,
                            ratio=C.RATIO, correct_t=C.CORRECT_T) -> pd.DataFrame:
    """Real pairs: illumination (i_*) vs viewpoint (v_*) with shipped H."""
    detectors = detectors or build_detectors()
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        for seq in seqs:
            for idx in idxs:
                img_ref, img_w, H, cat = load_hpatches_pair(seq, idx)
                kp_ref, kp_w, good_u = _match_images(
                    cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), det, norm, desc_obj, ratio)
                rows.append({"Cat": cat, "Seq": seq.split("/")[-1], "Pair": f"1->{idx}",
                             "Detector": det_name,
                             **evaluate_pair(kp_ref, kp_w, H,
                                             img_w.shape[:2], good_u, correct_t)})
    cols = ["Cat", "Seq", "Pair", "Detector", "n_features", "n_putative",
            "n_correct", "n_correspondences", "PMR", "Precision",
            "MatchingScore", "Recall"]
    return pd.DataFrame(rows)[cols]


def evaluate_photometric(base_image, detectors=None, ratio=C.RATIO,
                         correct_t=C.CORRECT_T) -> pd.DataFrame:
    """Synthetic illumination warps of the base image (H = I)."""
    detectors = detectors or build_detectors()
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    H_eye = np.eye(3)
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        kp_ref, des_ref = detect_and_describe(det, desc_obj, gray_base)
        for kind in C.PHOTO_KINDS:
            img_p = photo_warp(base_image, kind)
            kp_p, des_p = detect_and_describe(
                det, desc_obj, cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY))
            good_u = match_pair(kp_ref, des_ref, kp_p, des_p, norm, ratio)
            rows.append({"Cat": "illumination(synth)", "Warp": kind,
                         "Detector": det_name,
                         **evaluate_pair(kp_ref, kp_p, H_eye, img_p.shape,
                                         good_u, correct_t)})
    return pd.DataFrame(rows)


def _bench(fn, repeats=5):
    fn()  # warmup: faults pages in, stabilizes CPU caches/scaling
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(sorted(ts)[len(ts) // 2])


def _time_stages(det, desc_obj, norm, gray_ref, gray_tgt, repeats=5):
    """Time detect/describe on ref + target and the match step (ms, median).

    For joint detectors (SIFT/ORB/KAZE) describe is inseparable from detect,
    so ``t_describe_*`` is 0.0 and detect holds the joint cost. For FAST+*
    the two stages are timed separately. Returned matcher timings refer to
    the ``Matcher`` column value (classical configs: BFMatcher + ratio test;
    learned matchers add their own rows with the same schema).
    """
    if desc_obj is None:
        t_det_ref = _bench(lambda: det.detectAndCompute(gray_ref, None), repeats)
        t_det_tgt = _bench(lambda: det.detectAndCompute(gray_tgt, None), repeats)
        t_des_ref = t_des_tgt = 0.0
        _, des_ref = det.detectAndCompute(gray_ref, None)
        _, des_tgt = det.detectAndCompute(gray_tgt, None)
    else:
        t_det_ref = _bench(
            lambda: cv2.FastFeatureDetector_create().detect(gray_ref, None), repeats)
        fast = cv2.FastFeatureDetector_create()
        kp0 = fast.detect(gray_ref, None)
        for k in kp0:
            k.size = 31.0
            k.octave = 0
            k.angle = -1
        t_des_ref = _bench(lambda: desc_obj.compute(gray_ref, kp0), repeats)
        t_det_tgt = _bench(
            lambda: cv2.FastFeatureDetector_create().detect(gray_tgt, None), repeats)
        kp1 = fast.detect(gray_tgt, None)
        for k in kp1:
            k.size = 31.0
            k.octave = 0
            k.angle = -1
        t_des_tgt = _bench(lambda: desc_obj.compute(gray_tgt, kp1), repeats)
        _, des_ref = detect_and_describe(det, desc_obj, gray_ref)
        _, des_tgt = detect_and_describe(det, desc_obj, gray_tgt)
    bf = cv2.BFMatcher(norm, crossCheck=False)
    if (des_ref is not None and des_tgt is not None
            and len(des_ref) >= 2 and len(des_tgt) >= 2):
        t_match = _bench(lambda: bf.knnMatch(des_ref, des_tgt, k=2), repeats)
    else:
        t_match = 0.0
    return dict(t_detect_ref_ms=round(t_det_ref, 3),
                t_describe_ref_ms=round(t_des_ref, 3),
                t_detect_tgt_ms=round(t_det_tgt, 3),
                t_describe_tgt_ms=round(t_des_tgt, 3),
                t_match_ms=round(t_match, 3))


def benchmark_runtime(base_image, detectors=None, ajuntament_seq=C.REAL_SEQS[0],
                      ajuntament_idxs=C.REAL_IDXS, warps=None, repeats=5) -> pd.DataFrame:
    """Per-image, per-stage timing for every detector/descriptor/matcher.

    One row per (Detector, Matcher, Image): synthetic bird warps
    (Category ``viewpoint-synth``) plus real ajuntament pairs
    (Category ``illumination-real``). Stages are timed separately on the
    reference and target images, then the match step — all in ms, median of
    ``repeats`` runs, CPU. ``Matcher`` is ``BF-ratio<RATIO>`` for classical
    configs; learned matchers (SuperGlue/LightGlue) reuse this schema with
    their own matcher name and ``t_match_ms`` covering the learned forward.
    """
    from src.datasets import load_hpatches_pair  # local: avoid import cycle

    detectors = detectors or build_detectors()
    matcher = f"BF-ratio{C.RATIO}"
    images = build_synthetic_warps(base_image)
    if warps is not None:
        images = {k: v for k, v in images.items() if k in ("Original", *warps)}
    gray_ref = cv2.cvtColor(images["Original"], cv2.COLOR_BGR2GRAY)
    rows = []
    for det_name, (det, norm, desc_obj) in detectors.items():
        for warp_name, img_w in images.items():
            if warp_name == "Original":
                continue
            st = _time_stages(det, desc_obj, norm, gray_ref,
                              cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), repeats)
            rows.append({"Detector": det_name, "Matcher": matcher,
                         "Category": "viewpoint-synth", "Image": warp_name,
                         **st, "device": "cpu"})
        for idx in ajuntament_idxs:
            img_r, img_w, _, _ = load_hpatches_pair(ajuntament_seq, idx)
            st = _time_stages(det, desc_obj, norm,
                              cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY), repeats)
            rows.append({"Detector": det_name, "Matcher": matcher,
                         "Category": "illumination-real", "Image": f"1->{idx}",
                         **st, "device": "cpu"})
    df = pd.DataFrame(rows)
    df["t_total_ms"] = (df[[c for c in df.columns if c.startswith("t_")
                            and c.endswith("_ms")]].sum(axis=1).round(3))
    return df.sort_values(["Detector", "Category", "Image"]).reset_index(drop=True)


def descriptor_info_table(base_image, detectors=None) -> pd.DataFrame:
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    detectors = detectors or build_detectors()
    ref_data = {n: detect_and_describe(d, o, gray) for n, (d, _, o) in detectors.items()}
    return pd.DataFrame(descriptor_memory_table(ref_data)).sort_values("Detector")


# ---------------------------------------------------------------------------
# Learned path: SuperPoint+NN and SP+LightGlue (Part 4, merged from
# part3&4.py, minus SuperGlue). Scored with the same PMR/precision/recall
# code as the classical configs (TODO point 4).
# ---------------------------------------------------------------------------
def _learned_pair_metrics(n_features_sp, pts0, pts1, H, tgt_shape,
                          correct_t=C.CORRECT_T):
    """Full §2.8 metric dict for already-paired learned points.

    ``n_features_sp`` is the standalone SuperPoint detection count on the
    reference image (the PMR denominator, same basis as the SP+NN row);
    ``pts0``/``pts1`` are row-aligned matched points. Correctness is the
    same ≤``correct_t`` reprojection rule as ``evaluate_pair``.
    """
    from src.homography import project_points as _pp

    pts0 = np.asarray(pts0, dtype=np.float64).reshape(-1, 2)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    n_put = len(pts0)
    if n_features_sp == 0 or n_put == 0:
        return dict(n_features=n_features_sp, n_putative=n_put, n_correct=0,
                    n_correspondences=0, PMR=0.0, Precision=0.0,
                    MatchingScore=0.0, Recall=0.0)
    proj = _pp(H, pts0)
    hw, ww = tgt_shape[:2]
    inside = ((proj[:, 0] >= 0) & (proj[:, 0] < ww)
              & (proj[:, 1] >= 0) & (proj[:, 1] < hw))
    n_corr = int(inside.sum())
    n_ok = int((np.linalg.norm(proj - pts1, axis=1) <= correct_t).sum())
    return dict(
        n_features=n_features_sp, n_putative=n_put, n_correct=n_ok,
        n_correspondences=n_corr,
        PMR=n_put / n_features_sp if n_features_sp else 0.0,
        Precision=n_ok / n_put if n_put else 0.0,
        MatchingScore=n_ok / n_features_sp if n_features_sp else 0.0,
        Recall=n_ok / n_corr if n_corr else 0.0,
    )


def evaluate_learned_subset(base_image, ajuntament_seq=C.REAL_SEQS[0],
                            pair_idxs=(2, 6), warps=("Translated", "Rotated 45°"),
                            lg_threshold=0.1, device=None):
    """SuperPoint+NN vs SP+LightGlue on bird warps + ajuntament pairs.

    Returns a dict of DataFrames with the same schemas as the classical
    tables (``matching``, ``repeatability``, ``homography``, ``runtime``,
    ``descriptor``), using Detector ``SuperPoint+NN`` / ``SP+LightGlue`` and
    Matcher ``NN-crosscheck`` / ``LightGlue``. Raises ImportError with install
    instructions when ``transformers``/``torch``/``Pillow`` are missing.
    """
    from src.matching import (crosscheck_match, lightglue_match_hf,
                              superpoint_features_hf)

    try:
        import torch as _torch

        dev_default = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        dev_default = "cpu"
    device_eff = device or dev_default

    t0_all = time.perf_counter()
    images = build_synthetic_warps(base_image)
    images = {k: v for k, v in images.items() if k in ("Original", *warps)}
    gt = build_synthetic_gt(base_image.shape)

    # SuperPoint extraction is shared: time it per image, reuse everywhere.
    sp_cache: dict = {}

    def sp_feats(img_bgr, tag):
        if tag not in sp_cache:
            t0 = time.perf_counter()
            kxy, des = superpoint_features_hf(img_bgr, device=device_eff)
            sp_cache[tag] = (kxy, des, (time.perf_counter() - t0) * 1000.0)
        return sp_cache[tag]

    # (kind, label, tag, img_ref, img_tgt, H, tgt_shape) work items
    items = []
    for wn, img_w in images.items():
        if wn == "Original":
            continue
        items.append(("viewpoint-synth", wn, f"synth:{wn}",
                      images["Original"], img_w, gt[wn], img_w.shape))
    for idx in pair_idxs:
        img_r, img_w, H, _ = load_hpatches_pair(ajuntament_seq, idx)
        # ajuntament 1.ppm is identical for every pair: extract once
        items.append(("illumination-real", f"1->{idx}", f"aju:{idx}",
                      img_r, img_w, H, img_w.shape))
    m_rows, r_rows, h_rows, rt_rows = [], [], [], []
    desc_done = False
    for cat, label, tag, img_r, img_w, H, tshp in items:
        tag_r = "synth:Original" if cat == "viewpoint-synth" else "aju:ref"
        kxy_r, des_r, t_sp_r = sp_feats(img_r, tag_r)
        kxy_w, des_w, t_sp_w = sp_feats(img_w, tag)
        kp_r_cv, kp_w_cv = xy_to_keypoints(kxy_r), xy_to_keypoints(kxy_w)

        # --- SuperPoint + mutual-NN (cross-check, part3&4.py) ---
        t0 = time.perf_counter()
        good_nn = crosscheck_match(des_r, des_w, cv2.NORM_L2)
        t_nn = (time.perf_counter() - t0) * 1000.0
        pts0_nn = np.float32([kxy_r[m.queryIdx] for m in good_nn])
        pts1_nn = np.float32([kxy_w[m.trainIdx] for m in good_nn])
        print(f"SuperPoint+NN {label} -> "
              f"Matches={len(good_nn)} Precision="
              f"{(sum(1 for a, b in zip(pts0_nn, pts1_nn) if np.linalg.norm(project_points(H, a.reshape(1, 2))[0] - b) <= C.CORRECT_T) / len(good_nn)) if good_nn else 0.0:.3f}")
        m_rows.append({"Detector": "SuperPoint+NN", "Matcher": "NN-crosscheck",
                       "Category": cat, "Label": label,
                       **_learned_pair_metrics(len(kxy_r), pts0_nn, pts1_nn, H, tshp)})
        rep = detector_repeatability(kp_r_cv, kp_w_cv, H, tshp)
        r_rows.append({"Detector": "SuperPoint+NN", "Category": cat, "Label": label,
                       **rep})
        hm = homography_metrics(kp_r_cv, kp_w_cv, good_nn, H, C.RANSAC_T, tshp)
        hm.pop("H_est", None)
        h_rows.append({"Detector": "SuperPoint+NN", "Category": cat, "Label": label,
                       **hm})
        rt_rows.append({"Detector": "SuperPoint+NN", "Matcher": "NN-crosscheck",
                        "Category": cat, "Image": label,
                        "t_detect_ref_ms": round(t_sp_r, 3),
                        "t_describe_ref_ms": 0.0,
                        "t_detect_tgt_ms": round(t_sp_w, 3),
                        "t_describe_tgt_ms": 0.0,
                        "t_match_ms": round(t_nn, 3), "device": device_eff})

        # --- SuperPoint + LightGlue (part3&4.py, threshold=0.1) ---
        t0 = time.perf_counter()
        pts0_lg, pts1_lg = lightglue_match_hf(
            img_r, img_w, threshold=lg_threshold, device=device_eff)
        t_lg = (time.perf_counter() - t0) * 1000.0
        n_ok_lg = 0
        if len(pts0_lg):
            proj_lg = project_points(H, np.asarray(pts0_lg, dtype=np.float64))
            n_ok_lg = int((np.linalg.norm(
                proj_lg - np.asarray(pts1_lg, dtype=np.float64), axis=1)
                <= C.CORRECT_T).sum())
        print(f"LightGlue {label} -> Matches={len(pts0_lg)} Precision="
              f"{(n_ok_lg / len(pts0_lg)) if len(pts0_lg) else 0.0:.3f}")
        m_rows.append({"Detector": "SP+LightGlue", "Matcher": "LightGlue",
                       "Category": cat, "Label": label,
                       **_learned_pair_metrics(len(kxy_r), pts0_lg, pts1_lg, H, tshp)})
        kp0_lg, kp1_lg = xy_to_keypoints(pts0_lg), xy_to_keypoints(pts1_lg)
        good_lg = [cv2.DMatch(i, i, 0.0) for i in range(len(pts0_lg))]
        hm = homography_metrics(kp0_lg, kp1_lg, good_lg, H, C.RANSAC_T, tshp)
        hm.pop("H_est", None)
        h_rows.append({"Detector": "SP+LightGlue", "Category": cat, "Label": label,
                       **hm})
        rt_rows.append({"Detector": "SP+LightGlue", "Matcher": "LightGlue",
                        "Category": cat, "Image": label,
                        "t_detect_ref_ms": round(t_sp_r, 3),
                        "t_describe_ref_ms": 0.0,
                        "t_detect_tgt_ms": round(t_sp_w, 3),
                        "t_describe_tgt_ms": 0.0,
                        "t_match_ms": round(t_lg, 3), "device": device_eff})
        if not desc_done:
            desc_done = True
            desc_rows = descriptor_memory_table(
                {"SuperPoint+NN": (kp_r_cv, des_r)})
            d_df = pd.DataFrame(desc_rows)

    rt_df = pd.DataFrame(rt_rows)
    rt_df["t_total_ms"] = (rt_df[["t_detect_ref_ms", "t_describe_ref_ms",
                                  "t_detect_tgt_ms", "t_describe_tgt_ms",
                                  "t_match_ms"]].sum(axis=1).round(3))
    print(f"[learned] subset in {time.perf_counter() - t0_all:.1f}s")
    return {"matching": pd.DataFrame(m_rows), "repeatability": pd.DataFrame(r_rows),
            "homography": pd.DataFrame(h_rows), "runtime": rt_df,
            "descriptor": d_df}
