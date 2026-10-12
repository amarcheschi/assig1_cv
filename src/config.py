"""Shared constants for Assignment 1 (reproducibility, Assignment §7).

All thresholds / protocol values live here so every module and ``main.py``
uses the same numbers. Reference protocol: Heinly et al. ECCV 2012,
correctness radius 3px, Lowe ratio 0.8.
"""

# Matching / geometric verification (ref protocol, §2.8)
RATIO = 0.8          # ratio of two best candidate distances; reject if > 0.8
CORRECT_T = 3.0      # correspondence correct if reproj error <= 3.0 px
RANSAC_T = 3.0       # cv2.findHomography RANSAC reprojection threshold (px)
RANSAC_STUDY = (1.0, 3.0, 5.0)  # threshold variation for justification (§3.2)

# Synthetic geometric warps applied to the base image (§2.7 viewpoint part)
ROT_ANGLE = 45.0
TRANS_TX, TRANS_TY = 100, 50
SCALE_UP = 1.5
SCALE_DOWN = 0.5
COMBINED_ANGLE, COMBINED_SCALE = 30.0, 0.8
COMBINED_TX, COMBINED_TY = 50, 30

# Synthetic photometric warps (illumination part, H = I)
PHOTO_KINDS = ("bright+", "dark+", "contrast", "gamma0.5", "gamma2.0")

# Real HPatches illumination source: ONE sequence only (ajuntament).
# Viewpoint is evaluated on the bird image (synthetic warps), so no v_* here.
REAL_SEQS = ("hpatches-mini/i_ajuntament",)
REAL_IDXS = (2, 3, 4, 5, 6)  # all illumination levels of ajuntament 1.ppm
AJUNTAMENT_REF = "hpatches-mini/i_ajuntament/1.ppm"  # photometric base image

# FAST+BRIEF-family keypoint normalisation (so BRISK/FREAK are happy)
FAST_PATCH_SIZE = 31.0

# Stitching defaults (§3)
STITCH_RATIO = RATIO
STITCH_RANSAC_T = RANSAC_T
STITCH_CONFIGS = ("SIFT", "ORB", "FAST+BRIEF")
