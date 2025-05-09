import cv2
import numpy as np
import time
import traceback
from typing import Tuple, Optional, Dict, Any

from .base_estimator import BaseMotionEstimator

class SparseOpticalFlowEstimator(BaseMotionEstimator):
    """
    Estimates motion using Shi-Tomasi corner detection and
    Lucas-Kanade sparse optical flow. Refinds features when tracking fails
    """
    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)

        self._feature_params = dict(
            maxCorners=int(self._config.get('max_corners', 200)),
            qualityLevel=float(self._config.get('quality_level', 0.05)), # Lowered for potentially less textured ground
            minDistance=int(self._config.get('min_distance', 10)), # Increased slightly
            blockSize=int(self._config.get('block_size', 7))
        )

        self._lk_params = dict(
            winSize=(int(self._config.get('lk_window_size', 21)), int(self._config.get('lk_window_size', 21))), # Ensure tuple
            maxLevel=int(self._config.get('lk_pyramid_levels', 3)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      int(self._config.get('lk_max_iterations', 30)),
                      float(self._config.get('lk_epsilon', 0.01)))
        )

        self._min_tracked_points = int(self._config.get('min_tracked_points', 20)) # Min points to trust the flow

        if self._feature_params['maxCorners'] <= 0: raise ValueError("max_corners must be positive")
        if not (0 < self._feature_params['qualityLevel'] < 1): raise ValueError("qualityLevel must be between 0 and 1")
        if self._feature_params['minDistance'] <= 0: raise ValueError("minDistance must be positive")
        if self._lk_params['winSize'][0] % 2 == 0 or self._lk_params['winSize'][1] % 2 == 0 or self._lk_params['winSize'][0] <= 1:
             raise ValueError("lk_window_size must be an odd integer > 1")
        if self._lk_params['maxLevel'] < 0: raise ValueError("lk_pyramid_levels cannot be negative")
        if self._min_tracked_points < 3: raise ValueError("min_tracked_points should be at least 3")


        #self.logger.info(f"[{self.__class__.__name__}] Initialized.")
        #self.logger.info(f"  > Shi-Tomasi Params: {self._feature_params}")
        #self.logger.info(f"  > LK Params: {self._lk_params}")
        #self.logger.info(f"  > Min Tracked Points: {self._min_tracked_points}")

        # --- State for previous frame ---
        self.reset_state()

    def reset_state(self) -> None:
        #self.logger.debug(f"[{self.__class__.__name__}] Resetting state.")
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None # Shape (N, 1, 2)

    def _detect_features(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            points = cv2.goodFeaturesToTrack(gray_image, mask=None, **self._feature_params)
            if points is not None and len(points) > 0:
                #self.logger.debug(f"[{self.__class__.__name__}] Detected {len(points)} new features.")
                return points.astype(np.float32) # Ensure correct type for LK
            else:
                #self.logger.warn(f"[{self.__class__.__name__}] goodFeaturesToTrack found no points.")
                return None
        except Exception as e:
            #self.logger.error(f"[{self.__class__.__name__}] Error during feature detection: {e}")
            return None

    def estimate_displacement(self, current_frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        start_time = time.monotonic()
        logger_prefix = f"[{self.__class__.__name__}]"

        if current_frame_bgr is None or current_frame_bgr.size == 0:
            #self.logger.warn(f"{logger_prefix} Received invalid current frame.")
            return None, None

        try:
            current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            median_dx = None
            median_dy = None
            flow_time = 0.0

            # --- Check if we can track from the previous frame ---
            if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
                start_flow_time = time.monotonic()
                # Calculate optical flow
                next_points, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, current_gray, self.prev_points, None, **self._lk_params
                )
                flow_time = (time.monotonic() - start_flow_time) * 1000

                # Select good points (status == 1)
                if next_points is not None and status is not None:
                    good_mask = status.ravel() == 1
                    good_prev_points = self.prev_points[good_mask]
                    good_next_points = next_points[good_mask]
                    num_tracked = len(good_prev_points)

                    #self.logger.debug(f"{logger_prefix} Tracked {num_tracked} points ({flow_time:.1f}ms).")

                    # --- Check Minimum Tracked Points & Calculate Displacement ---
                    if num_tracked >= self._min_tracked_points:
                        displacements = good_next_points - good_prev_points
                        if displacements.size > 0:
                             # Ensure shape is (N, 2) before median
                            if displacements.ndim == 3 and displacements.shape[1] == 1:
                                displacements = displacements.reshape(-1, 2)

                            if displacements.ndim == 2 and displacements.shape[1] == 2:
                                median_dx = np.median(displacements[:, 0])
                                median_dy = np.median(displacements[:, 1])
                                #self.logger.debug(f"{logger_prefix} Calculated median disp: dx={median_dx:.2f}, dy={median_dy:.2f}")
                                # Update points for the *next* iteration
                                self.prev_points = good_next_points.reshape(-1, 1, 2)
                            else:
                                #self.logger.warning(f"{logger_prefix} Invalid displacements shape after filtering: {displacements.shape}")
                                self.prev_points = None # Force feature redetection
                        else:
                            #self.logger.warn(f"{logger_prefix} Displacement calculation resulted in empty array.")
                            self.prev_points = None # Force feature redetection
                    else:
                        #self.logger.warn(f"{logger_prefix} Insufficient tracked points ({num_tracked} < {self._min_tracked_points}). Redetecting.")
                        self.prev_points = None # Force feature redetection
                else:
                    #self.logger.warn(f"{logger_prefix} Optical flow calculation failed (returned None). Redetecting.")
                    self.prev_points = None # Force feature redetection

            # --- Detect new features if needed (first frame or tracking failed) ---
            if self.prev_points is None:
                #self.logger.debug(f"{logger_prefix} Detecting new features for tracking.")
                start_detect_time = time.monotonic()
                self.prev_points = self._detect_features(current_gray)
                detect_time = (time.monotonic() - start_detect_time) * 1000
                #self.logger.debug(f"{logger_prefix} Feature detection time: {detect_time:.1f}ms")
                # No displacement can be calculated when redetecting
                median_dx = None
                median_dy = None

            # Update previous frame state
            self.prev_gray = current_gray

            total_time = (time.monotonic() - start_time) * 1000
            #if median_dx is not None:
                 #self.logger.debug(f"{logger_prefix} Estimation complete. Total time: {total_time:.1f}ms")
            #else:
                 #self.logger.debug(f"{logger_prefix} Estimation yielded no displacement this frame. Total time: {total_time:.1f}ms")

            return float(median_dx) if median_dx is not None else None, \
                   float(median_dy) if median_dy is not None else None

        except Exception as e:
            error_trace = traceback.format_exc()
            #self.logger.error(f"{logger_prefix} Unexpected error during displacement estimation: {e}\n{error_trace}")
            self.reset_state() 
            return None, None