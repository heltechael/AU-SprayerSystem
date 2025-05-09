# /mnt/c/Users/Mic45/Documents/AU/jetson_simulation/orei-cam_ws/src/vision_system/vision_system/motion_estimation/orb_homography_estimator.py

import cv2
import numpy as np
import time
import traceback
from typing import Tuple, Optional, Dict, Any

from .base_estimator import BaseMotionEstimator

class OrbHomographyEstimator(BaseMotionEstimator):
    """
    Estimates motion using ORB features, KNN matching, and RANSAC homography
    """
    # Hardcoded IGNORE masks - if image contains static elements (e.g. wheel)
    IGNORE_TOP_PERCENT = 0.1
    IGNORE_BOTTOM_PERCENT = 0.0
    IGNORE_RECT = (400, 0, 900, 750) # (x_min, y_min, x_max, y_max)

    # ------------------------------------

    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)

        self._num_features = int(self._config.get('num_features', 2000))
        self._knn_k = int(self._config.get('knn_k', 2))
        self._lowe_ratio = float(self._config.get('lowe_ratio', 0.75))
        self._ransac_threshold = float(self._config.get('ransac_threshold', 5.0))
        self._min_good_matches = int(self._config.get('min_good_matches', 10))

        if self._num_features <= 0: raise ValueError("num_features must be positive")
        if self._knn_k < 2: raise ValueError("knn_k must be at least 2 for ratio test")
        if not (0 < self._lowe_ratio < 1): raise ValueError("lowe_ratio must be between 0 and 1")
        if self._ransac_threshold <= 0: raise ValueError("ransac_threshold must be positive")
        if self._min_good_matches <= 4: raise ValueError("min_good_matches should be > 4 for Homography")

        try:
            self.orb = cv2.ORB_create(nfeatures=self._num_features)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.logger.info(f"[{self.__class__.__name__}] Initialized ORB(nfeatures={self._num_features}) and BFMatcher(HAMMING)")
            self.logger.info(f"  > Matching Params: k={self._knn_k}, ratio={self._lowe_ratio}")
            self.logger.info(f"  > RANSAC Params: threshold={self._ransac_threshold}, min_matches={self._min_good_matches}")
            self.logger.info(f"  > Detection Masking: Ignoring top {self.IGNORE_TOP_PERCENT*100:.1f}% of image rows.")

        except Exception as e:
            self.logger.fatal(f"[{self.__class__.__name__}] Failed to initialize OpenCV components: {e}")
            raise

        self.reset_state() 

    def reset_state(self) -> None:
        """Resets the previous frame state and the detection mask."""
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[Tuple[cv2.KeyPoint]] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self._detection_mask: Optional[np.ndarray] = None

    def _create_or_update_mask(self, image_shape: Tuple[int, int]) -> None:
        h, w = image_shape
        create_new = False
        if self._detection_mask is None:
            create_new = True
            #self.logger.info(f"[{self.__class__.__name__}] Creating initial detection mask for size {w}x{h}.")
        elif self._detection_mask.shape != image_shape:
            create_new = True
            self.logger.warning(f"[{self.__class__.__name__}] Image shape changed from {self._detection_mask.shape[::-1]} to {w}x{h}. Recreating detection mask.")

        if create_new:
            self._detection_mask = np.ones((h, w), dtype=np.uint8) * 255

            # Ignore top rows
            if self.IGNORE_TOP_PERCENT > 0:
                ignore_rows = int(h * self.IGNORE_TOP_PERCENT)
                if ignore_rows > 0:
                    self._detection_mask[0:ignore_rows, :] = 0 # Set top rows to 0 (ignore)
                    self.logger.info(f"  > Mask: Ignored top {ignore_rows} rows.")

            # Ignore bottom rows
            if hasattr(self, 'IGNORE_BOTTOM_PERCENT') and self.IGNORE_BOTTOM_PERCENT > 0:
                ignore_rows = int(h * self.IGNORE_BOTTOM_PERCENT)
                if ignore_rows > 0 and h - ignore_rows < h:
                    self._detection_mask[h-ignore_rows:h, :] = 0
                    self.logger.info(f"  > Mask: Ignored bottom {ignore_rows} rows.")

            # Ignore rectangle
            if hasattr(self, 'IGNORE_RECT'):
                x1, y1, x2, y2 = self.IGNORE_RECT
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                if x2_c > x1_c and y2_c > y1_c:
                    self._detection_mask[y1_c:y2_c, x1_c:x2_c] = 0
                    self.logger.info(f"  > Mask: Ignored rect ({x1_c}, {y1_c}) to ({x2_c}, {y2_c}).")

    def estimate_displacement(self, current_frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimates the (dx, dy) displacement using ORB/Homography with masking
        """
        start_time = time.monotonic()
        logger_prefix = f"[{self.__class__.__name__}]"

        if current_frame_bgr is None or current_frame_bgr.size == 0:
            return None, None

        try:
            current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

            self._create_or_update_mask(current_gray.shape)

            detect_start_time = time.monotonic()
            current_keypoints, current_descriptors = self.orb.detectAndCompute(
                current_gray, mask=self._detection_mask
            )
            detect_time = (time.monotonic() - detect_start_time) * 1000

            num_detected = len(current_keypoints) if current_keypoints is not None else 0
            # self.logger.debug(f"{logger_prefix} Detected {num_detected} features in allowed regions ({detect_time:.1f}ms).")

            if current_keypoints is None or current_descriptors is None or num_detected == 0:
                # self.logger.warn(f"{logger_prefix} No keypoints/descriptors found in current frame (after masking).")
                self.reset_state() 
                self.prev_gray = current_gray 
                return None, None

            if self.prev_gray is None or self.prev_keypoints is None or self.prev_descriptors is None:
                # self.logger.info(f"{logger_prefix} First frame processed, storing features for next frame.")
                self.prev_gray = current_gray
                self.prev_keypoints = current_keypoints
                self.prev_descriptors = current_descriptors
                return None, None # No displacement on the first frame

            # Feature matching (KNN) 
            start_match_time = time.monotonic()
            matches = self.bf_matcher.knnMatch(self.prev_descriptors, current_descriptors, k=self._knn_k)
            match_time = (time.monotonic() - start_match_time) * 1000

            # Ratio test (Lowe's Test) 
            good_matches = []
            if matches:
                for match_pair in matches:
                    if len(match_pair) == self._knn_k:
                        m, n = match_pair
                        if m.distance < self._lowe_ratio * n.distance:
                            # Check indices validity 
                            if m.queryIdx < len(self.prev_keypoints) and m.trainIdx < len(current_keypoints):
                                good_matches.append(m)
                            else:
                                self.logger.warning(f"{logger_prefix} Invalid index in good match, skipping.")
            else:
                self.logger.warn(f"{logger_prefix} KNN matching returned no matches.")

            num_good_matches = len(good_matches)
            # self.logger.debug(f"{logger_prefix} Found {num_good_matches} good matches after ratio test.")

            # Check min matches and estimate homography 
            if num_good_matches < self._min_good_matches:
                # self.logger.warn(f"{logger_prefix} Not enough good matches ({num_good_matches} < {self._min_good_matches}) to estimate motion.")
                self.prev_gray = current_gray
                self.prev_keypoints = current_keypoints
                self.prev_descriptors = current_descriptors
                return None, None

            # extract locations of good matches
            prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC
            start_ransac_time = time.monotonic()
            H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, self._ransac_threshold)
            ransac_time = (time.monotonic() - start_ransac_time) * 1000

            if H is None or mask is None:
                # self.logger.warn(f"{logger_prefix} RANSAC Homography estimation failed.")
                self.prev_gray = current_gray
                self.prev_keypoints = current_keypoints
                self.prev_descriptors = current_descriptors
                return None, None

            # calc avg displacement from RANSAC inliers 
            inlier_mask = mask.ravel() == 1
            num_inliers = np.sum(inlier_mask)

            if num_inliers < self._min_good_matches:
                 # self.logger.warn(f"{logger_prefix} Not enough RANSAC inliers ({num_inliers} < {self._min_good_matches}) after Homography.")
                 self.prev_gray = current_gray
                 self.prev_keypoints = current_keypoints
                 self.prev_descriptors = current_descriptors
                 return None, None

            prev_inliers = prev_pts[inlier_mask]
            curr_inliers = curr_pts[inlier_mask]
            displacements = curr_inliers - prev_inliers

            if displacements.size == 0: 
                 self.logger.warn(f"{logger_prefix} Displacements array is empty after RANSAC.")
                 self.prev_gray = current_gray
                 self.prev_keypoints = current_keypoints
                 self.prev_descriptors = current_descriptors
                 return None, None

            median_dx = np.median(displacements[:, 0, 0])
            median_dy = np.median(displacements[:, 0, 1])

            total_time = (time.monotonic() - start_time) * 1000
            self.prev_gray = current_gray
            self.prev_keypoints = current_keypoints
            self.prev_descriptors = current_descriptors

            return float(median_dx), float(median_dy)

        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"{logger_prefix} Unexpected error during displacement estimation: {e}\n{error_trace}")
            self.reset_state() 
            return None, None