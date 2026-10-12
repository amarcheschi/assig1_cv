"""Visualizations: keypoints, matches, correct/incorrect, bar charts (§5)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_correct_incorrect(img_ref, kp_ref, img_w, kp_w, ok, bad,
                           title, save=None, show=False):
    """Green correct vs red incorrect matches side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mlist, color, tag in zip(
            axes, [ok, bad], [(0, 255, 0), (255, 0, 0)], ["correct", "incorrect"]):
        vis = cv2.drawMatches(img_ref, kp_ref, img_w, kp_w, mlist, None,
                              matchColor=color,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title} {tag} ({len(mlist)})")
        ax.axis("off")
    fig.tight_layout()
    _save(fig, Path(save) if save else None)
    if show:
        plt.show()
    plt.close(fig)


def plot_matches_grid(img_ref, images, results, save=None, show=False):
    """Rows = detector, cols = warp: putative match grids."""
    det_names = list(results)
    warps = [w for w in next(iter(results.values()))]
    fig, axes = plt.subplots(len(det_names), len(warps),
                             figsize=(5 * len(warps), 4 * len(det_names)),
                             squeeze=False)
    for i, det_name in enumerate(det_names):
        for j, warp_name in enumerate(warps):
            n_good, _, kp_ref, kp_w, good = results[det_name][warp_name]
            vis = cv2.drawMatches(img_ref, kp_ref, images[warp_name], kp_w, good,
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            axes[i][j].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[i][j].set_title(f"{det_name} — {warp_name}\n{n_good} good matches")
            axes[i][j].axis("off")
    fig.tight_layout()
    _save(fig, Path(save) if save else None)
    if show:
        plt.show()
    plt.close(fig)


def plot_keypoints_panels(img_bgr, kp_dict, title="Keypoints per detector",
                          save=None, show=False):
    """One panel per detector: rich keypoints on the same image."""
    names = list(kp_dict)
    n = len(names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             squeeze=False)
    for i, name in enumerate(names):
        kps = kp_dict[name]
        vis = cv2.drawKeypoints(
            img_bgr, kps, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax = axes[i // cols][i % cols]
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name} ({len(kps)} kps)")
        ax.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, Path(save) if save else None)
    if show:
        plt.show()
    plt.close(fig)


def plot_bar(df, x_col, y_col, hue_col, title, ylabel, save=None, show=False):
    import pandas as pd  # noqa: F401 (kept local: plotting-only dep pattern)

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = sorted(df[x_col].unique())
    hues = list(df[hue_col].unique())
    x = np.arange(len(xs))
    width = 0.8 / max(len(hues), 1)
    for k, h in enumerate(hues):
        sub = df[df[hue_col] == h].set_index(x_col)
        ax.bar(x + k * width, [sub.loc[v, y_col] if v in sub.index else 0 for v in xs],
               width, label=h)
    ax.set_xticks(x + width * (len(hues) - 1) / 2)
    ax.set_xticklabels(xs, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(save) if save else None)
    if show:
        plt.show()
    plt.close(fig)
