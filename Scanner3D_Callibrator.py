"""Camera calibration for PS4 Eye.

Fixes vs original:
- robust handling when no images are found
- clear errors when chessboard detection fails on all images

Expected workflow:
  1) Run TakeImages.py (saves to ./pics/*.jpg)
  2) Run this script from the same folder

Outputs:
  - param_ret.npy
  - param_K.npy
  - param_dist.npy
"""

from __future__ import annotations

import glob

import cv2
import numpy as np


def main() -> None:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard with 9x6 inner corners.
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob("calib_droite/*.png"))
    if not images:
        raise FileNotFoundError(
            "No images found in ./pics/*.jpg. Run TakeImages.py first (it saves to ./pics/)."
        )

    last_gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] Failed to read {fname}, skipping")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"[INFO] Detected chessboard corners in {fname}") # Debug log to confirm detection
        else:
            print(f"[WARN] Chessboard not found in {fname}")

    if not objpoints or last_gray_shape is None:
        raise RuntimeError(
            "No valid calibration images: chessboard corners were not detected. "
            "Try better lighting, sharper focus, and more varied angles." 
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_gray_shape, None, None)
    print(K)

    with open("param_ret.npy", "wb") as f:
        np.save(f, ret)

    with open("param_K.npy", "wb") as f:
        np.save(f, K)

    with open("param_dist.npy", "wb") as f:
        np.save(f, dist)

    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += float(error)

    print("total error:", mean_error / max(1, len(objpoints)))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
