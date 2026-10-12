"""Homography utilities: GT warps, projection, RANSAC estimation (§3.2)."""

from __future__ import annotations

import cv2
import numpy as np

from src import config as C


def to_h3(M2x3) -> np.ndarray:
    """Promote a 2x3 affine matrix to a 3x3 homography."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = np.asarray(M2x3, dtype=np.float64)
    return H


def build_synthetic_gt(shape) -> dict:
    """Exact 3x3 homographies ORIGINAL -> each synthetic warp.

    Mirrors exactly how the warps are produced in ``datasets.build_synthetic_warps``.
    """
    h, w = shape[:2]
    center = (w / 2, h / 2)
    H_rot = to_h3(cv2.getRotationMatrix2D(center, C.ROT_ANGLE, 1.0))
    H_trans = to_h3(np.float32([[1, 0, C.TRANS_TX], [0, 1, C.TRANS_TY]]))

    sh, sw = int(round(h * C.SCALE_UP)), int(round(w * C.SCALE_UP))
    x0, y0 = (sw - w) // 2, (sh - h) // 2
    H_s15 = np.array([[C.SCALE_UP, 0, -x0], [0, C.SCALE_UP, -y0], [0, 0, 1.]])

    sh2, sw2 = int(round(h * C.SCALE_DOWN)), int(round(w * C.SCALE_DOWN))
    px, py = (w - sw2) // 2, (h - sh2) // 2
    H_s05 = np.array([[C.SCALE_DOWN, 0, px], [0, C.SCALE_DOWN, py], [0, 0, 1.]])

    M_c = cv2.getRotationMatrix2D(center, C.COMBINED_ANGLE, C.COMBINED_SCALE)
    M_c[0, 2] += C.COMBINED_TX
    M_c[1, 2] += C.COMBINED_TY
    return {
        "Original": np.eye(3),
        "Rotated 45°": H_rot,
        "Translated": H_trans,
        "Scaled 1.5x": H_s15,
        "Scaled 0.5x": H_s05,
        "Combined warp": to_h3(M_c),
    }


def project_points(H, pts_xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N,2) array of (x, y) points."""
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def homography_metrics(kp_ref, kp_w, good_u, H_gt, ransac_thresh: float = C.RANSAC_T,
                       img_shape=None) -> dict:
    """RANSAC homography + quality stats (§3.1 step 7, §3.2).

    Returns n_matches / n_inliers / inlier_ratio / mean+median residual of
    inliers (``||H_est·p1 − p2||``) / corner-mean error vs GT.
    """
    if len(good_u) < 4:
        return dict(n_matches=len(good_u), n_inliers=0, inlier_ratio=0.0,
                    mean_resid=np.nan, med_resid=np.nan, corner_err=np.nan,
                    H_est=None)
    p1 = np.float32([kp_ref[m.queryIdx].pt for m in good_u]).reshape(-1, 1, 2)
    p2 = np.float32([kp_w[m.trainIdx].pt for m in good_u]).reshape(-1, 1, 2)
    H_est, mask = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_thresh)
    if H_est is None or mask is None:
        return dict(n_matches=len(good_u), n_inliers=0, inlier_ratio=0.0,
                    mean_resid=np.nan, med_resid=np.nan, corner_err=np.nan,
                    H_est=None)
    inl = mask.ravel() == 1
    n_in = int(inl.sum())
    proj = cv2.perspectiveTransform(p1[inl], H_est)
    errs = np.linalg.norm((proj - p2[inl]).reshape(-1, 2), axis=1)
    if img_shape is not None:
        hh, ww = img_shape[:2]
    else:
        ww = hh = 1000  # corner error is relative; GT comparison dominates
    corners = np.float32([[[0, 0]], [[ww, 0]], [[ww, hh]], [[0, hh]]])
    ce = float(np.linalg.norm(
        (cv2.perspectiveTransform(corners, H_est)
         - cv2.perspectiveTransform(corners, np.asarray(H_gt, float))).reshape(-1, 2),
        axis=1).mean())
    return dict(n_matches=len(good_u), n_inliers=n_in,
                inlier_ratio=n_in / len(good_u),
                mean_resid=float(errs.mean()), med_resid=float(np.median(errs)),
                corner_err=ce, H_est=H_est)
