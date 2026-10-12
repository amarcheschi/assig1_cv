# Assignment 1 — Report Handover Notes (Part 1 only)

For the colleague writing the final report. Covers: code structure, what each
module does, the exact experimental protocol, all results with analysis,
methodology details (warmup, median, timing schema), and known limitations.
**Stitching (Part 2) is not implemented — this document covers Part 1 only.**

Status: TODO.md points 1–2 (metrics, illumination vs viewpoint) done and
reproduced from modules; points 3–4 (custom BRIEF ×2 strategies, SuperPoint+NN,
SP+LightGlue) merged from `part3&4.py` and evaluated in the same protocol.

---

## 1. Project structure

```
Assig1/
  main.py                  # Part 1 entry point. Full pipeline, runnable.
  requirements.txt         # incl. transformers + Pillow for the learned path
  part3&4.py               # SOURCE material (already merged — do not run; see §7)
  TODO.md                  # task list with per-point status
  src/
    config.py              # ALL thresholds/constants in one place (§7 repro)
    detectors.py           # detector factory: SIFT/ORB/KAZE + FAST+* (8 configs)
    descriptors.py         # OpenCV descriptors + custom BRIEF (uniform/gaussian)
    matching.py            # ratio test, one-to-one, cross-check, learned matchers
    homography.py          # GT warps, projection, RANSAC estimation/quality
    datasets.py            # synthetic warps, photometric warps, HPatches loading
    evaluation.py          # metrics + all experiment drivers
    visualization.py       # all figures
  results/                 # 8 CSVs/JSON tables (the report's data source)
  figures/                 # 20 PNGs (the report's figure source)
  hpatches-sequences-release/  # viewpoint source (v_bird used)
  hpatches-mini/               # illumination source (i_ajuntament used)
```

### What each module does (with key functions)

- **`src/config.py`** — single source of truth: `RATIO=0.8`, `CORRECT_T=3.0px`,
  `RANSAC_T=3.0` (+ study values 1/3/5), warp parameters, `REAL_SEQS`,
  `REAL_IDXS=(2,3,4,5,6)`, `PHOTO_KINDS`, FAST patch size. Cite these in the
  report instead of scattering numbers.
- **`src/detectors.py`** — `build_detectors()` returns the 8 configs as
  `(det, norm, desc_obj)` triples; `detect_and_describe()` is the shared
  detect/describe routine (FAST keypoints get fixed size/octave, no
  orientation, so BRISK/FREAK behave deterministically).
- **`src/descriptors.py`** — `opencv_build_info()` (documents the 4.13.0.92
  build + BRIEF/BRISK/FREAK availability, SURF skipped as optional);
  `descriptor_memory_table()`; Part 3 merge: `generate_uniform_pairs`,
  `generate_gaussian_pairs`, `compute_brief`, `bits_to_bytes` (exact port of
  `part3&4.py`, plus a `seed` for reproducibility — the file used unseeded
  `np.random`), and `CustomBrief` delegating to them (+ Gaussian smoothing
  per BRIEF paper §3).
- **`src/matching.py`** — `ratio_match` (kNN k=2 + Lowe 0.8), `unique_matches`
  (greedy one-to-one, required by the §2.8 protocol), `match_pair`,
  `split_correct`, `crosscheck_match` (mutual-NN, used for custom BRIEF and
  SuperPoint+NN exactly as in `part3&4.py`), `xy_to_keypoints`; learned:
  `superpoint_features_hf`, `lightglue_match_hf`, `run_lightglue_pair`
  (HF `magic-leap-community/superpoint` + `ETH-CVG/lightglue_superpoint`,
  threshold 0.1, process-wide model cache), legacy `run_superglue_pair`
  (Magicleap repo, kept, not evaluated).
- **`src/homography.py`** — `to_h3`, `build_synthetic_gt` (exact homographies
  matching how the warps are rendered), `project_points`,
  `homography_metrics` (RANSAC H + inliers/ratio + mean/median inlier residual
  + corner-mean error vs GT).
- **`src/datasets.py`** — `build_synthetic_warps` (Rotated 45°, Translated
  (100,50), Scaled 1.5x/0.5x, Combined 30°+0.8+shift), `photo_warp`
  (bright+/dark+/contrast/gamma0.5/gamma2.0, geometry = identity),
  `load_hpatches_pair` (real pair + shipped `H_1_k` + auto
  illumination/viewpoint label).
- **`src/evaluation.py`** — `evaluate_pair` (PMR/Precision/MatchingScore/Recall),
  `detector_repeatability` (detector-only), drivers `evaluate_synthetic`,
  `evaluate_repeatability_synthetic`, `evaluate_homography_synthetic`,
  `evaluate_real_hpatches`, `evaluate_photometric`, `benchmark_runtime`
  (per-image per-stage, see §3), `descriptor_info_table`,
  `evaluate_learned_subset` (SP+NN vs LightGlue on Translated/Rotated +
  ajuntament 1→2/1→6, same schemas).
- **`src/visualization.py`** — `plot_correct_incorrect` (green correct vs
  **blue** incorrect — `(255,0,0)` is BGR blue, a known cosmetic bug; read
  blue as "wrong"), `plot_keypoints_panels`, `plot_matches_grid`, `plot_bar`.
- **`main.py`** — `run_part1()` orchestrates everything; flags `--quick`
  (SIFT+ORB smoke test), `--learned` (opt-in learned subset, needs
  transformers/torch/Pillow + one-time model download), `--show`, `--device`.

### How to run
- `python main.py` — full classical (8 configs), ~20 min CPU.
- `python main.py --learned` — full + learned rows appended, ~25–45 min CPU.
- `python main.py --quick [--learned]` — smoke test (~1 min).
- Install: `pip install -r requirements.txt` (torch 2.14 CPU verified;
  transformers 5.19 verified; both HF model IDs verified to exist).

---

## 2. Experimental protocol (for Methods section)

- **Viewpoint**: 5 synthetic warps of `hpatches-sequences-release/v_bird/1.ppm`
  (973×1068) with exact GT homographies. Translated is a sanity check (pure
  shift → Precision ≈ 1.0 for all); ranking signal comes from
  Rotated/Scaled/Combined.
- **Illumination**: (a) real `hpatches-mini/i_ajuntament` pairs 1→2…1→6
  (mild→severe, shipped H); (b) 5 synthetic photometric warps of
  `ajuntament/1.ppm` (480×640) with H=I, isolating tone-mapping effects.
- **Matching**: ratio 0.8 + one-to-one enforcement everywhere (classical);
  cross-check for custom BRIEF and SP+NN (file parity); LightGlue threshold 0.1.
  Correctness radius 3px; RANSAC 3px justified by the 1/3/5 study printed to
  console (1px too strict under warp interpolation, 5px adds nothing).
- **Metrics** (§2.8): `PMR=putative/features`, `Precision=correct/putative`,
  `MatchingScore=correct/features (=PMR×Precision)`,
  `Recall=correct/correspondences`. Every table carries its denominators
  (`n_features/n_putative/n_correct/n_correspondences`) so values are
  reproducible. For LightGlue, PMR's denominator is the standalone
  SuperPoint detection count on the reference image (documented assumption in
  `evaluate_learned_subset`, since LightGlue only outputs paired points).

---

## 3. Timing methodology (read before citing any ms value)

- Each stage is reported as **median of 5 runs + 1 discarded warmup**
  (`evaluation._bench`). Warmup absorbs cold-start effects: CPU frequency
  ramp, page faults/cache fill, lazy thread-pool/BLAS init. Without it, the
  first (slow) sample would corrupt the median — most visible on ~20ms stages
  like ORB detect.
- Table columns are **per-single-execution** costs: `t_detect/describe`
  timed separately on ref and target images (joint `detectAndCompute` methods
  report describe = 0.0 by construction), plus `t_match`; `t_total_ms` is their
  sum. `Matcher` column is `BF-ratio0.8` classically, `NN-crosscheck` /
  `LightGlue` for learned (match time = NN search / learned forward).
- **Do NOT sum the column as run wall-clock**: the benchmark executes each
  stage 6× (warmup+5), so benchmarking alone costs ≈6× the column sum; the
  ~24 min full run is dominated by that plus untimed eval passes,
  visualization recompute, learned inference, and figure I/O.
- **LibreOffice locale trap**: CSVs store dots as decimals (`120.859` ms).
  Italian-locale Calc parses `.` as thousands separator and shows `120859`.
  Import with English (US) locale. Values are milliseconds.

---

## 4. Results analysis (for Results/Discussion section)

### 4.1 Synthetic viewpoint (`notebook_matching_metrics.csv`, 44 rows)
- **SIFT is the stable all-rounder**: Precision 0.958–0.998 on every warp
  (e.g. Rotated 1954/2006 = 97.4%, Translated 2903/2910).
- **BRIEF has no rotation invariance**: OpenCV 0/253, custom-uniform 1/169,
  custom-gaussian 2/376 correct on Rotated 45° — three independent
  implementations failing identically is strong evidence, and the
  detector/descriptor split (same FAST points work at Translated: 6429/6429).
- **Custom vs OpenCV BRIEF**: near-identical behavior (aju 1→2 scores
  0.32/0.27 vs 0.35); uniform slightly more precise on average than gaussian
  (0.59 vs 0.45 over illumination levels) — the genuine ≥2-strategy
  comparison §2.5 asks for.
- **BRISK/FREAK rotate but don't scale**: Rotated precision ≈0.97/0.95
  (4014/3129 correct) but Scaled 0.5x gives 28/36 putative, 0 correct.
- **ORB/KAZE middle**: ORB precision 0.93–0.98 everywhere (note the 5000-point
  cap); KAZE strong on rotation/combined (scores 0.63/0.36).
- **Learned**: Translated tied (SP+NN 0.857, LightGlue 0.855); Rotated splits
  them (NN 0.216 vs LightGlue 0.770) — same features, matcher decides. Honest
  note: LightGlue rotation (0.77) still trails SIFT (0.97) on synthetic warps;
  learned earns it back on real illumination (below).

### 4.2 Repeatability (`notebook_repeatability.csv`, detector-only)
ORB highest (0.93–0.98), FAST variants 0.77–0.94, SIFT lowest on some warps
(0.61–0.72) — yet SIFT matches best. Headline: **repeatability ≠ matching
quality**; never rank by keypoint count (7363 FAST vs 3361 SIFT).

### 4.3 Homography (`notebook_homography.csv`)
Healthy rows: inlier ratio ≈1.0, residuals <1px, corner error <1px (all SIFT
≤0.32px). Garbage-H signature: single-digit inliers + hundred-px corner_err
(BRIEF/Rotated 6 inl/847px; BRISK/Scaled-0.5x 5/715px; FREAK/Scaled-0.5x
6/417px) — RANSAC "consensus" among outliers; such H would destroy a panorama.

### 4.4 Real illumination (`notebook_illum_ajuntament.csv`)
Cliff after 1→2 for everyone: scores 0.27–0.56 (KAZE best 0.559, ORB 0.384,
SIFT 0.344) → <0.04 at 1→3 → ~0.01 at 1→4…6. PMR and precision die together
(SIFT PMR 0.40→0.04). Qualitative companion: `real_easy_SIFT.png` (50/50
horizontal lines, H≈I) vs `real_hard_SIFT.png` (7 survivors on stable
structures, 26 chaotic lines, day-vs-night). Note the easy-pair wrong matches
are mostly *horizontal* too — repetitive facade windows fool descriptors even
with trivial geometry (report-grade observation).

### 4.5 Synthetic photometric (`notebook_illum_synth.csv`, H=I)
`contrast` ≈ free (0.95–1.0: order-preserving stretch); `dark+` hardest
(PMR halves: SIFT 0.186, ORB 0.207); **precision stays ~0.93–1.0 throughout**
— tone mapping removes matches without corrupting survivors. Contrast with the
real cliff (both PMR *and* precision die): real illumination ≠ global tone
curve.

### 4.6 Cost (`notebook_descriptor_info.csv`, `notebook_runtime.csv`)
Memory: binary 32B/des (ORB/BRIEF 160–240KB) vs SIFT 512B (1.7MB), KAZE 256B
(1.6MB), SuperPoint 1024B (1.2MB). Time: KAZE dominates (~1200ms/bird pair,
~330ms/aju); SIFT ~300/~80; ORB ~85/~20; FAST detect ~1–3ms, BRIEF describe
~13ms vs BRISK ~58ms; learned on CPU: SP extraction ~7s bird/~1s aju per
image, NN match ~15ms, LightGlue forward ~2–3s. Image-size effect: bird costs
~3–4× ajuntament everywhere. Verdict: **ORB for real-time, SIFT for accuracy**,
KAZE accurate but ~4× SIFT.

### 4.7 Figures → report mapping (§5)
`keypoints_original.png` (viz per method); `correct_incorrect_*.png` ×10
(correct/incorrect examples, incl. empty BRIEF panel + 2 learned);
`matches_grid.png` + `match_survival.png` (Table 1 companions);
`runtime_split/per_image.png` (Table 2); `real_illum_curve.png` +
`real_easy/hard` (illumination story); `repeatability.png`;
`photo_warps.png` (methods illustration). Homography quality lives only in
the CSV — quote garbage-H rows in text.

---

## 5. Limitations to state honestly (Discussion/Conclusion)
1. Unequal keypoint budgets (ORB capped 5000, FAST ~7000 free, SIFT 3361) —
   `detectors.ipynb` BLOCK 8 matched-~1000 analysis exists for this.
2. One-directional repeatability (ref→target over `n_valid`), optimistic for
   dense detectors — flatters ORB.
3. Translated is trivially easy; affine-only synthetic warps (no perspective,
   occlusion, blur); one base image per protocol (bird/ajuntament).
4. RANSAC study is SIFT-only; LightGlue PMR denominator is a documented
   assumption; incorrect-match lines render blue (BGR tuple bug).
5. Learned runs are CPU-timed; GPU would change absolute (not relative) costs.
6. **Do NOT mix in `part3&4.py`'s printed numbers** (540 matches/21.9%,
   SP+NN 92.6%/26.8%, LightGlue 87.6%) — those are `Data/graf` pairs
   (folder absent here), superseded by our protocol tables above.

## 6. Open TODO for the report writer
- Missing: Part 2 stitching pipeline (no `stitching.py`; `cv2.Stitcher` must
  NOT be the solution), SIFT+LightGlue variant (§2.6 allows it), SURF note is
  written (skipped/optional) — just cite it, and the blue-vs-red one-line fix
  in `visualization.py` if figures are re-rendered.
