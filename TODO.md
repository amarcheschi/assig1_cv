# TODO — Assignment 1: Keypoints, Descriptors, Stitching

Status (2026-10-07):  `Readme.txt` is empty.

Ref: `Assignment_1_Keypoints_Descriptors_Stitching.pdf` §2–§8.

## 1. Part 1 metrics still missing in notebook (§2.7–§2.8)

Notebook `CVC_SuperPoint+SuperGlue.ipynb` currently computes only:
`n_features / n_putative / n_correct / n_correspondences / PMR / Precision /
MatchingScore / Recall` on synthetic warps.

- [ ] Fix threshold inconsistency: matching cell uses `RATIO=0.75`, eval cell
  labels `RATIO=0.8` but never re-matches (it reuses `results` + `unique_matches`).
  Pick one (ref protocol `0.8`), document `CORRECT_T=3.0px`, re-run.
- [ ] Repeatability in CVC notebook (detector-only, no descriptor):
  `project_points(H) → in-bounds → nearest-tgt <3px / n_correspondences`.
  `detectors.ipynb` BLOCKs 4–5 had this; re-run and link, don't duplicate silently.
- [ ] Number + percentage correct + correct/incorrect visualization.
  The green/red `drawMatches` code (cells 14–15) is commented out — restore for
  ≥1 pair per warp.
- [ ] Homography estimation: `cv2.findHomography(p1,p2,cv2.RANSAC,3.0)` per
  detector×warp. Report `H_est`, corner-mean error vs `GT_HOMOGRAPHIES`,
  `n_inliers`, `inlier_ratio`, mean/median `||H_est·p1−p2||` on inliers.
  Justify the RANSAC threshold + vary it once (e.g. 1/3/5px).
- [ ] Runtime split: `perf_counter()` around detect / describe / match separately
  (now only whole-pipeline timing existed in deleted baseline). State CPU/GPU.
- [ ] Descriptor dim/memory: `notebook_descriptor_info.csv` added — extend with
  `psutil` process delta if used, and SuperPoint (256×float32) rows once added.

## 2. Illumination vs viewpoint — at least 2 categories (§2.7)

CVC notebook = only synthetic geometric warps of `v_bird/1.ppm`.
No light changes, no real HPatches `H_1_*`.

- [ ] Real pairs from `hpatches-mini/i_*` (illumination: `i_ajuntament,i_dome,
  i_lionday,i_veggies,i_whitebuilding`) vs `v_*` (viewpoint) using shipped
  `1.ppm + H_1_2…H_1_6`. Same `evaluate_pair`/`unique_matches` protocol.
- [ ] Synthetic photometric warps for `v_bird/1.ppm`: brightness/contrast/gamma
  (e.g. `alpha/beta`, `gamma=0.5/2.0`), then repeat matching table split
  illumination vs viewpoint. This is the "lights rather than perspective" gap.
- [ ] Tables split `illumination / viewpoint`, with denominators
  (`n_features`, `n_correspondences`) so recall is reproducible.

## 3. Binary descriptors (§2.5)

- [ ] Custom simplified BRIEF with ≥2 sampling strategies (e.g. uniform-random
  vs Gaussian pairs). Currently only `cv2.xfeatures2d.BriefDescriptorExtractor`.
- [ ] Document exact OpenCV build (`4.13.0.92`, `opencv-contrib`) + which of
  BRIEF/BRISK/FREAK/SURF was available. SURF optional — state explicitly if skipped.

## 4. Learned methods (§2.6) — mostly commented out

- [ ] SuperPoint + NN matching, scored with the same PMR/precision/recall code.
- [ ] SuperPoint + SuperGlue: restore demo cells (weights `indoor/outdoor`,
  `match_threshold`, `sinkhorn_iterations`, device), record checkpoint/repo.
- [ ] SuperPoint/SIFT + LightGlue: missing entirely — add, incl. early-stop/pruning
  note. Keep detector/descriptor vs matcher performance separate in text.

## 5. Part 2 stitching pipeline (§3) — not started

No `stitching.py` / `homography.py`; `cv2.warpPerspective` only in synthetic test.

- [ ] Steps 1–12 as functions: load (color+gray) → detect → describe (norm per
  descriptor: L2 vs Hamming) → match + ratio/cross-check → RANSAC `findHomography`
  → report matches/inliers/ratio/reprojection error → viz keypoints + inliers →
  `warpPerspective` canvas → average-blend baseline (+ optional feather/multiband).
- [ ] ≥3 configs incl. 1 learned (e.g. SIFT, ORB, FAST+BRIEF, KAZE, SP+LightGlue).
- [ ] 3-image panorama + reference comparison (central vs side) + chained-H
  explanation and error-accumulation note.

## 6. Report inputs (§5) + comparisons (§4)

- [ ] Keypoint viz per method; correct/incorrect match examples.
- [ ] Table 1: repeatability/matching (split illum/viewpoint).
- [ ] Table 2: runtime + cost (detect/describe/match, dim/memory).
- [ ] Homography quality (inliers + error) + panoramas per method.
- [ ] Discussion: why each method wins/fails per transform; accuracy-vs-efficiency;
  real-time pick vs accuracy pick. Don't rank by keypoint count alone.

## 7. Reproducibility (§7–§8)

- [ ] Fill `Readme.txt` → `README.md`: env creation, HPatches placement
  (`hpatches-mini/`, `hpatches-sequences-release/`), one-command repro,
  CPU/GPU note. `requirements.txt` already OK.
- [ ] Modularize from notebooks → `src/{detectors,descriptors,matching,
  homography,stitching,evaluation,visualization}.py` + `main.py` (notebooks stay
  as demos). Record all thresholds (FAST, ratio, correctness, RANSAC).
- [ ] Cite SuperPoint/SuperGlue/LightGlue impl. + checkpoints; submission zip
  `LastNames_Assign1.zip` with Code/README/requirements, data instructions,
  outputs, Report (intro / methods 3–5pp / setup+results 2–5pp / conclusion).
