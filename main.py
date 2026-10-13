"""Part 1 entry point: detectors/descriptors/matching evaluation (§2).

Reproduces the CVC_SuperPoint+SuperGlue notebook (points 1–2 of TODO.md)
plus part3&4.py (points 3–4: custom BRIEF both strategies, SuperPoint+NN,
SP+LightGlue) as callable modules:

    python main.py [--quick] [--learned] [--base PATH] [--outdir DIR] [--show]

Outputs (results/): matching metrics, repeatability, homography quality,
per-image per-stage runtime, descriptor memory, ajuntament illumination
pairs + synthetic photometric (ajuntament-based) tables, run config.
Figures go to figures/. Viewpoint = bird image only; illumination =
ajuntament sequence only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.datasets import build_synthetic_warps, load_color, photo_warp
from src.descriptors import opencv_build_info
from src.detectors import build_detectors, detect_and_describe
from src.evaluation import (
    descriptor_info_table,
    detector_repeatability,
    evaluate_homography_synthetic,
    evaluate_learned_subset,
    evaluate_pair,
    evaluate_photometric,
    evaluate_real_hpatches,
    evaluate_repeatability_synthetic,
    evaluate_synthetic,
    benchmark_runtime,
)
from src.homography import build_synthetic_gt, homography_metrics
from src.matching import match_pair, split_correct, unique_matches
from src.visualization import (
    plot_bar,
    plot_correct_incorrect,
    plot_keypoints_panels,
    plot_matches_grid,
)


def _det_subset(detectors, names):
    return {n: detectors[n] for n in names} if names else detectors


def run_part1(base_path: str, outdir: Path, figdir: Path, quick: bool,
              show: bool, detectors=None, learned: bool = False,
              device=None):
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    base = load_color(base_path)
    print(f"[base] {base_path} shape={base.shape}")
    base_aju = load_color(C.AJUNTAMENT_REF)
    print(f"[illum-base] {C.AJUNTAMENT_REF} shape={base_aju.shape}")
    print(f"[opencv] {opencv_build_info()}")

    detectors = detectors or build_detectors()
    if quick:
        # smoke test: 2 detectors, 2 warps, fewer real pairs
        detectors = _det_subset(detectors, ["SIFT", "ORB"])
        warps = ["Translated", "Rotated 45°"]
    else:
        warps = None

    # illumination protocol: ajuntament sequence only (viewpoint = bird only)
    seqs = list(C.REAL_SEQS)
    idxs = (2,) if quick else C.REAL_IDXS

    # --- synthetic viewpoint matching (RATIO=0.8, CORRECT_T=3.0px) ---
    df_m = evaluate_synthetic(base, detectors, warps=warps)
    df_m.to_csv(outdir / "notebook_matching_metrics.csv", index=False)
    print(f"[matching] wrote {outdir/'notebook_matching_metrics.csv'} ({len(df_m)} rows)")

    # --- repeatability (detector-only) ---
    df_r = evaluate_repeatability_synthetic(base, detectors, warps=warps)
    df_r.to_csv(outdir / "notebook_repeatability.csv", index=False)
    print(f"[repeatability] wrote {outdir/'notebook_repeatability.csv'}")

    # --- homography + RANSAC threshold study ---
    df_h = evaluate_homography_synthetic(base, detectors)
    if quick:
        df_h = df_h[df_h["Warp"].isin(["Translated", "Rotated 45°"])]
    df_h.to_csv(outdir / "notebook_homography.csv", index=False)
    print(f"[homography] wrote {outdir/'notebook_homography.csv'}")
    # 1/3/5px study on SIFT Translated + Rotated (if present)
    images = build_synthetic_warps(base)
    gt = build_synthetic_gt(base.shape)
    gray_ref = cv2.cvtColor(images["Original"], cv2.COLOR_BGR2GRAY)
    if "SIFT" in detectors:
        det, norm, dobj = detectors["SIFT"]
        for wn in ["Translated", "Rotated 45°"]:
            if wn not in images:
                continue
            kp_ref, des_ref = detect_and_describe(det, dobj, gray_ref)
            kp_w, des_w = detect_and_describe(
                det, dobj, cv2.cvtColor(images[wn], cv2.COLOR_BGR2GRAY))
            gu = match_pair(kp_ref, des_ref, kp_w, des_w, norm)
            for t in C.RANSAC_STUDY:
                m = homography_metrics(kp_ref, kp_w, gu, gt[wn], t, images[wn].shape)
                print(f"SIFT {wn} RANSAC={t}: inliers={m['n_inliers']}/{m['n_matches']} "
                      f"ratio={m['inlier_ratio']:.3f} mean_resid={m['mean_resid']:.3f} "
                      f"corner_err={m['corner_err']:.3f}")

    # --- runtime: per-image, per-stage, per detector/descriptor/matcher ---
    df_rt = benchmark_runtime(base, detectors, ajuntament_idxs=idxs, warps=warps)
    df_rt.to_csv(outdir / "notebook_runtime.csv", index=False)
    print(f"[runtime] wrote {outdir/'notebook_runtime.csv'}")
    df_d = descriptor_info_table(base, detectors)
    df_d.to_csv(outdir / "notebook_descriptor_info.csv", index=False)
    print(f"[descriptors]\n{df_d.to_string(index=False)}")

    # --- run config (reproducibility §7) ---
    cfg = {
        "base_image": str(base_path),
        "matching_ratio": C.RATIO,
        "correct_thresh_px": C.CORRECT_T,
        "ransac_thresh_px": C.RANSAC_T,
        "ransac_study": list(C.RANSAC_STUDY),
        "one_to_one": True,
        "opencv_version": cv2.__version__,
        "opencv_build": opencv_build_info(),
        "image_shape_hw": list(base.shape[:2]),
        "warps": sorted([w for w in images if w != "Original"]),
        "homographies": {k: v.tolist() for k, v in gt.items()},
        "illumination_seq": list(C.REAL_SEQS),
        "illumination_idxs": list(idxs),
        "photometric_base": C.AJUNTAMENT_REF,
        "runtime_schema": ["Detector", "Matcher", "Category", "Image",
                           "t_detect_ref_ms", "t_describe_ref_ms",
                           "t_detect_tgt_ms", "t_describe_tgt_ms",
                           "t_match_ms", "t_total_ms", "device"],
        # placeholder: overwritten below once the opt-in learned block runs
        "learned": {"enabled": False},
    }
    (outdir / "notebook_run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[config] wrote {outdir/'notebook_run_config.json'}")

    # --- real HPatches: ajuntament illumination levels only ---
    df_real = evaluate_real_hpatches(detectors, seqs=seqs, idxs=idxs)
    df_real.to_csv(outdir / "notebook_illum_ajuntament.csv", index=False)
    print(f"[real] wrote {outdir/'notebook_illum_ajuntament.csv'}")
    print(df_real[["Pair", "Detector", "n_putative", "n_correct",
                   "PMR", "Precision", "MatchingScore", "Recall"]]
          .to_string(index=False))
    print("[real] mean over illumination levels:")
    print(df_real.groupby("Detector")[["PMR", "Precision", "MatchingScore",
                                       "Recall"]].mean().round(4).to_string())

    # --- synthetic photometric warps of the ajuntament base (H = I) ---
    df_ph = evaluate_photometric(base_aju, detectors)
    df_ph.to_csv(outdir / "notebook_illum_synth.csv", index=False)
    print(f"[photometric] wrote {outdir/'notebook_illum_synth.csv'}")

    # --- learned path, opt-in (Part 4: SuperPoint+NN, SP+LightGlue) ---
    learned_info = {"enabled": False}
    if learned:
        try:
            pair_ids = tuple(dict.fromkeys((idxs[0], idxs[-1])))
            learned_frames = evaluate_learned_subset(
                base, pair_idxs=pair_ids,
                warps=tuple(warps) if warps else ("Translated", "Rotated 45°"),
                device=device)
            lm = learned_frames["matching"]
            lm_synth = lm[lm["Category"] == "viewpoint-synth"].rename(
                columns={"Label": "Warp"})
            lm_real = lm[lm["Category"] == "illumination-real"]
            df_m = pd.concat([df_m, lm_synth[[
                "Detector", "Warp", "n_features", "n_putative", "n_correct",
                "n_correspondences", "PMR", "Precision", "MatchingScore",
                "Recall"]]], ignore_index=True)
            df_real = pd.concat([df_real, lm_real.assign(
                Cat="illumination", Seq="i_ajuntament").rename(
                columns={"Label": "Pair"})[[
                "Cat", "Seq", "Pair", "Detector", "n_features", "n_putative",
                "n_correct", "n_correspondences", "PMR", "Precision",
                "MatchingScore", "Recall"]]], ignore_index=True)
            lr = learned_frames["repeatability"]
            lr_synth = lr[lr["Category"] == "viewpoint-synth"].rename(
                columns={"Label": "Warp"})
            df_r = pd.concat([df_r, lr_synth[[
                "Detector", "Warp", "n_ref", "n_valid", "n_repeatable",
                "repeatability"]]], ignore_index=True)
            lh = learned_frames["homography"]
            lh["Warp"] = lh["Label"]  # synth warps + "1->k" aju pairs, one table
            df_h = pd.concat([df_h, lh[[
                "Detector", "Warp", "n_matches", "n_inliers", "inlier_ratio",
                "mean_resid", "med_resid", "corner_err"]]], ignore_index=True)
            df_rt = pd.concat(
                [df_rt, learned_frames["runtime"]], ignore_index=True)
            df_d = pd.concat(
                [df_d, learned_frames["descriptor"]], ignore_index=True)
            for _name, _df in [
                    ("notebook_matching_metrics.csv", df_m),
                    ("notebook_illum_ajuntament.csv", df_real),
                    ("notebook_repeatability.csv", df_r),
                    ("notebook_homography.csv", df_h),
                    ("notebook_runtime.csv", df_rt),
                    ("notebook_descriptor_info.csv", df_d)]:
                _df.to_csv(outdir / _name, index=False)
            print("[learned] appended SuperPoint+NN / SP+LightGlue rows "
                  "to all tables")
            learned_info = {"enabled": True,
                            "device": learned_frames["runtime"]["device"].iloc[0],
                            "sp_model": "magic-leap-community/superpoint",
                            "lg_model": "ETH-CVG/lightglue_superpoint",
                            "lg_threshold": 0.1,
                            "warps": sorted(lm_synth["Warp"].unique().tolist()),
                            "pairs": sorted(lm_real["Label"].unique().tolist())}
        except ImportError as exc:
            print(f"[learned] skipped: {exc}")
    # record the learned outcome in the already-written run config
    cfg_path = outdir / "notebook_run_config.json"
    cfg_saved = json.loads(cfg_path.read_text())
    cfg_saved["learned"] = learned_info
    cfg_path.write_text(json.dumps(cfg_saved, indent=2))

    # --- visualizations (§5: keypoints, correct/incorrect, tables-as-charts) ---
    gray_w_rot = None
    if "Rotated 45°" in images:
        gray_w_rot = cv2.cvtColor(images["Rotated 45°"], cv2.COLOR_BGR2GRAY)
        # correct/incorrect for EVERY detector on Rotated 45° (not just 2)
        for det_name in detectors:
            det, norm, dobj = detectors[det_name]
            kp_ref, des_ref = detect_and_describe(det, dobj, gray_ref)
            kp_w, des_w = detect_and_describe(det, dobj, gray_w_rot)
            gu = match_pair(kp_ref, des_ref, kp_w, des_w, norm)
            ok, bad = split_correct(kp_ref, kp_w, gt["Rotated 45°"], gu)
            pct = len(ok) / len(gu) * 100 if gu else 0.0
            print(f"{det_name} Rotated 45°: {len(ok)}/{len(gu)} correct ({pct:.1f}%)")
            plot_correct_incorrect(
                images["Original"], kp_ref, images["Rotated 45°"], kp_w,
                ok, bad, f"{det_name} Rotated 45°",
                save=figdir / f"correct_incorrect_{det_name.replace('+', '_')}.png",
                show=show)

        # learned correct/incorrect on the same pair (reuses SP extraction)
        if learned_info.get("enabled"):
            from src.matching import (crosscheck_match, lightglue_match_hf,
                                      superpoint_features_hf, xy_to_keypoints)
            kxy_r, des_r = superpoint_features_hf(
                images["Original"], device=device)
            kxy_w, des_w = superpoint_features_hf(
                images["Rotated 45°"], device=device)
            kp_r, kp_w = xy_to_keypoints(kxy_r), xy_to_keypoints(kxy_w)
            good_nn = crosscheck_match(des_r, des_w, cv2.NORM_L2)
            ok, bad = split_correct(kp_r, kp_w, gt["Rotated 45°"], good_nn)
            print(f"SuperPoint+NN Rotated 45°: {len(ok)}/{len(good_nn)} correct")
            plot_correct_incorrect(
                images["Original"], kp_r, images["Rotated 45°"], kp_w,
                ok, bad, "SuperPoint+NN Rotated 45°",
                save=figdir / "correct_incorrect_SuperPoint_NN.png", show=show)
            pts0, pts1 = lightglue_match_hf(
                images["Original"], images["Rotated 45°"], device=device)
            kp0_lg, kp1_lg = xy_to_keypoints(pts0), xy_to_keypoints(pts1)
            good_lg = [cv2.DMatch(i, i, 0.0) for i in range(len(pts0))]
            ok, bad = split_correct(kp0_lg, kp1_lg, gt["Rotated 45°"], good_lg)
            print(f"SP+LightGlue Rotated 45°: {len(ok)}/{len(good_lg)} correct")
            plot_correct_incorrect(
                images["Original"], kp0_lg, images["Rotated 45°"], kp1_lg,
                ok, bad, "SP+LightGlue Rotated 45°",
                save=figdir / "correct_incorrect_SP_LightGlue.png", show=show)

    # keypoints per detector on the base image (§5: keypoint viz per method)
    kp_panels = {}
    for det_name in detectors:
        det, norm, dobj = detectors[det_name]
        kp, _ = detect_and_describe(det, dobj, gray_ref)
        kp_panels[det_name] = kp
    plot_keypoints_panels(images["Original"], kp_panels,
                          save=figdir / "keypoints_original.png", show=show)

    # full putative-match grid, rows=detector cols=warp (recomputed once)
    match_results, match_images = {}, {}
    for det_name in detectors:
        det, norm, dobj = detectors[det_name]
        kp_ref, des_ref = detect_and_describe(det, dobj, gray_ref)
        match_results[det_name] = {}
        for warp_name, img_w in images.items():
            if warp_name == "Original":
                continue
            match_images[warp_name] = img_w
            kp_w, des_w = detect_and_describe(
                det, dobj, cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY))
            good, n_raw = match_pair(kp_ref, des_ref, kp_w, des_w, norm), 0
            # match_pair already returns one-to-one list; keep count pair
            match_results[det_name][warp_name] = (
                len(good), n_raw, kp_ref, kp_w, good)
    plot_matches_grid(images["Original"], match_images, match_results,
                      save=figdir / "matches_grid.png", show=show)

    # --- bar charts from the saved tables (no recompute) ---
    plot_bar(df_m, "Warp", "n_correct", "Detector", "Match survival across warps",
             "Correct matches", save=figdir / "match_survival.png", show=show)
    plot_bar(df_r, "Warp", "repeatability", "Detector",
             "Repeatability across warps", "Repeatability",
             save=figdir / "repeatability.png", show=show)
    stage_cols = ["t_detect_ref_ms", "t_describe_ref_ms", "t_detect_tgt_ms",
                  "t_describe_tgt_ms", "t_match_ms"]
    rt_mean = df_rt.groupby("Detector")[stage_cols].mean().reset_index()
    rt_long = rt_mean.melt(id_vars=["Detector"], value_vars=stage_cols,
                           var_name="Stage", value_name="ms")
    plot_bar(rt_long, "Stage", "ms", "Detector",
             "Mean stage time per detector (median of 5, over all images)",
             "ms", save=figdir / "runtime_split.png", show=show)
    plot_bar(df_rt, "Image", "t_total_ms", "Detector",
             "Total time per image (detect ref+tgt + describe ref+tgt + match)",
             "ms", save=figdir / "runtime_per_image.png", show=show)
    plot_bar(df_real, "Pair", "MatchingScore", "Detector",
             "Ajuntament illumination difficulty curve (MatchingScore)",
             "MatchingScore", save=figdir / "real_illum_curve.png", show=show)

    # real-pair match examples: easy vs hard ajuntament levels (§5)
    from src.datasets import load_hpatches_pair as _load_pair

    for idx, tag in [(idxs[0], "easy"), (idxs[-1], "hard")]:
        seq = seqs[0]
        img_r, img_w, H_r, _ = _load_pair(seq, idx)
        g_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        g_w = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)
        det, norm, dobj = detectors.get("SIFT", next(iter(detectors.values())))
        det_name = "SIFT" if "SIFT" in detectors else next(iter(detectors))
        kp_r, des_r = detect_and_describe(det, dobj, g_r)
        kp_w2, des_w = detect_and_describe(det, dobj, g_w)
        gu = match_pair(kp_r, des_r, kp_w2, des_w, norm)
        ok, bad = split_correct(kp_r, kp_w2, H_r, gu)
        plot_correct_incorrect(
            img_r, kp_r, img_w, kp_w2, ok[:50], bad[:50],
            f"{det_name} {seq.split('/')[-1]} 1->{idx} (top 50)",
            save=figdir / f"real_{tag}_{det_name}.png", show=show)

    # photometric warp strip of the ajuntament base + 5 illumination warps
    strip = [base_aju] + [photo_warp(base_aju, k) for k in C.PHOTO_KINDS]
    fig, axes = plt.subplots(1, len(strip), figsize=(4 * len(strip), 4))
    for ax, im, name in zip(axes, strip, ["Original", *C.PHOTO_KINDS]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figdir / "photo_warps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] Part 1 in {time.perf_counter() - t_all:.1f}s. "
          f"Points 1–2 of TODO.md reproduced from src/ modules.")


def main():
    ap = argparse.ArgumentParser(description="Assignment 1 — Part 1 pipeline")
    ap.add_argument("--base", default="hpatches-sequences-release/v_bird/1.ppm",
                    help="base image for synthetic warps")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--quick", action="store_true", help="smoke test subset")
    ap.add_argument("--show", action="store_true", help="display figures")
    ap.add_argument("--learned", action="store_true",
                    help="also run SuperPoint+NN and SP+LightGlue subset "
                         "(needs transformers/torch/Pillow + model download)")
    ap.add_argument("--device", default=None,
                    help="torch device for learned path (default: auto)")
    a = ap.parse_args()
    run_part1(a.base, Path(a.outdir), Path(a.figdir), a.quick, a.show,
              learned=a.learned, device=a.device)


if __name__ == "__main__":
    main()
