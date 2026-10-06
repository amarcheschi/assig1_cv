import cv2, numpy as np, random
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import cv2 as cv

image_path = "Data/graf/img1.ppm"

# read images and homography matrix
image = cv.imread(image_path, 1) # 0: grayscale | 1: bgr

h, w = image.shape[:2]
center = (w / 2, h / 2)

# ---------------------------
# 1. Rotation (e.g., 45 degrees)
# ---------------------------
angle = 45
scale_rot = 1.0
M_rot = cv.getRotationMatrix2D(center, angle, scale_rot)
img_rotated = cv.warpAffine(image, M_rot, (w, h))

# ---------------------------
# 2. Translation (shift by tx, ty)
# ---------------------------
tx, ty = 100, 50
M_trans = np.float32([[1, 0, tx],
                      [0, 1, ty]])
img_translated = cv.warpAffine(image, M_trans, (w, h))

# ---------------------------
# 3. Scale (resize by factor, then crop/pad back to original size)
# ---------------------------
scale_factor = 1.5
img_scaled_full = cv.resize(image, None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)
# Crop back to original size (center crop)
sh, sw = img_scaled_full.shape[:2]
x0 = (sw - w) // 2
y0 = (sh - h) // 2
img_scaled = img_scaled_full[y0:y0 + h, x0:x0 + w]

scale_factor = 0.5
img_scaled_full2 = cv.resize(image, None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)
# Crop back to original size (center crop)
sh2, sw2 = img_scaled_full2.shape[:2]
x2 = (sw2 - w) // 2
y2 = (sh2 - h) // 2
img_scaled2 = img_scaled_full2[y2:y2 + h, x2:x2 + w]

# ---------------------------
# 4. Combined warp (rotation + translation + scale in one affine matrix)
# ---------------------------
angle_c = 30
scale_c = 0.8
M_rot_c = cv.getRotationMatrix2D(center, angle_c, scale_c)
M_rot_c[0, 2] += 50   # add translation x
M_rot_c[1, 2] += 30   # add translation y
img_combined = cv.warpAffine(image, M_rot_c, (w, h))

# ---------------------------
# Run FAST on each version
# ---------------------------
fast = cv.FastFeatureDetector_create()

def run_fast(img, use_nms=True):
    fast.setNonmaxSuppression(use_nms)

    kps = fast.detect(img, None)

    vis = cv.drawKeypoints(
        img,
        kps,
        None,
        color=(255, 0, 0)
    )

    return {
        "image": img,
        "keypoints": kps,
        "n_keypoints": len(kps),
        "visualization": vis
    }

images = {
    "Original":       image,
    "Rotated 45°":    img_rotated,
    "Translated":     img_translated,
    "Scaled 1.5x":    img_scaled,
    "Scaled 0.5x":    img_scaled2,
    "Combined warp":  img_combined,
}

results = {}

for name, im in images.items():

    fast_nms = run_fast(im, use_nms=True)
    fast_no  = run_fast(im, use_nms=False)

    results[name] = {
        "fast_nms": fast_nms,
        "fast_no_nms": fast_no
    }

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

for name, data in results.items():

    img = data["fast_nms"]["image"]
    kps = data["fast_nms"]["keypoints"]

    kps_brief, des_brief = brief.compute(img, kps)

    data["brief"] = {
        "keypoints": kps_brief,
        "descriptors": des_brief
    }

"""
brisk = cv.BRISK_create()

for name, data in results.items():

    img = data["fast_nms"]["image"]

    kps_brisk, des_brisk = brisk.detectAndCompute(
        img,
        None
    )

    data["brisk"] = {
        "keypoints": kps_brisk,
        "descriptors": des_brisk,
        "n_keypoints": len(kps_brisk)
    }

BRISK doesn't work in my OpenCV build
"""

freak = cv2.xfeatures2d.FREAK_create()

for name, data in results.items():

    img = data["fast_nms"]["image"]
    kps = data["fast_nms"]["keypoints"]

    kps_freak, des_freak = freak.compute(img, kps)

    data["freak"] = {
        "keypoints": kps_freak,
        "descriptors": des_freak
    }

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
brief_ref = results["Original"]["brief"]["descriptors"]
freak_ref = results["Original"]["freak"]["descriptors"]

matching_results = []

for name, data in results.items():

    brief_desc = data["brief"]["descriptors"]
    freak_desc = data["freak"]["descriptors"]

    # BRIEF
    brief_matches = bf.match(brief_ref, brief_desc)
    brief_matches = sorted(brief_matches,
                           key=lambda x: x.distance)

    # FREAK
    freak_matches = bf.match(freak_ref, freak_desc)
    freak_matches = sorted(freak_matches,
                           key=lambda x: x.distance)

    matching_results.append({
        "Transform": name,
        "BRIEF matches": len(brief_matches),
        "BRIEF avg dist":
            np.mean([m.distance for m in brief_matches]),
        "FREAK matches": len(freak_matches),
        "FREAK avg dist":
            np.mean([m.distance for m in freak_matches])
    })


import pandas as pd

df_matches = pd.DataFrame(matching_results)

print(df_matches)

"""
In this experiment with this Image FREAK has obtain better results than BRIEF in all the transformations.
"""