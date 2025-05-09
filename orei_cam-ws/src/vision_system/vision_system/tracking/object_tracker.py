import time
import traceback
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment # For Hungarian algorithm matching

from ..common.definitions import Detection, TrackedObject

INF_COST = 1e5

class ObjectTracker:
    """
    Optimized object tracker using spatial pruning and pre-computation,
    based on experimental matching logic with external motion input.
    """
    def __init__(self, config: Dict[str, Any], image_dims: Tuple[int, int], logger):
        """
        Initializes the Optimized ObjectTracker.

        Args:
            config: Dictionary containing tracking parameters (tracking subsection).
            image_dims: Tuple (width, height) of the processed images.
            logger: An rclpy logger instance.
        """
        self._config = config
        self.logger = logger

        self._tracking_enabled = self._config.get('enabled', True)
        self._max_age = self._config.get('max_age', 5)
        self._min_confidence_new_track = self._config.get('min_confidence_for_new_track', 0.3)
        self._edge_threshold_px = self._config.get('close_to_edge_threshold_px', 10)
        self._bbox_touching_edge_weight = self._config.get('bbox_touching_edge_weight', 0.25)
        self._lock_crops_with_psez = self._config.get('lock_beets_with_psez', True) # Key from config

        self._vert_tolerance_px = self._config.get('vertical_match_tolerance_px', 80.0)
        self._horiz_tolerance_px = self._config.get('horizontal_match_tolerance_px', 100.0)
        self._validation_tolerance_factor = self._config.get('validation_tolerance_factor', 1.5)
        self._distance_penalty_threshold_factor = self._config.get('distance_penalty_threshold_factor', 2.0)

        self._search_radius_x = self._horiz_tolerance_px * self._validation_tolerance_factor
        self._search_radius_y = self._vert_tolerance_px * self._validation_tolerance_factor

        weights = self._config.get('weight_costs', {})
        self._pos_weight = weights.get('position', 1.0)
        self._area_weight = weights.get('area', 0.5)
        self._aspect_weight = weights.get('aspect', 0.5)
        self._class_weight = weights.get('class_consistency', 0.8)
        self._horiz_weight = weights.get('horizontal', 0.3)

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_track_id: int = 1
        self._image_width: int = image_dims[0]
        self._image_height: int = image_dims[1]
        self.finalize_callback = None

        self.crop_class_id: Optional[int] = None
        self.psez_class_id: Optional[int] = None

        if not self._tracking_enabled:
            self.logger.info(f"[{self.__class__.__name__}] Tracking is DISABLED.")
            return 

        self.logger.info(f"[{self.__class__.__name__}] Initialized (Optimized).")
        self.logger.info(f"  > Params: Max Age={self._max_age}, Min Conf={self._min_confidence_new_track}, LockCrop={self._lock_crops_with_psez}")
        self.logger.info(f"  > Tolerances (px): Vert={self._vert_tolerance_px:.1f}, Horiz={self._horiz_tolerance_px:.1f}")
        self.logger.info(f"  > Search Radii (px): Vert={self._search_radius_y:.1f}, Horiz={self._search_radius_x:.1f}")
        self.logger.info(f"  > Cost Weights: Pos={self._pos_weight:.2f}, Area={self._area_weight:.2f}, Aspect={self._aspect_weight:.2f}, Class={self._class_weight:.2f}, Horiz={self._horiz_weight:.2f}")

    def set_class_ids(self, crop_id: int, psez_id: int):
        if not isinstance(crop_id, int) or crop_id < 0: raise ValueError("Invalid crop_id")
        if not isinstance(psez_id, int) or psez_id < 0: raise ValueError("Invalid psez_id")
        self.crop_class_id = crop_id
        self.psez_class_id = psez_id
        #self.logger.info(f"[{self.__class__.__name__}] Class IDs set: Crop={crop_id}, PSEZ={psez_id}")

    def update(self,
               detections: List[Detection],
               motion_dx: Optional[float],
               motion_dy: Optional[float]
               ) -> List[TrackedObject]:
        if not self._tracking_enabled:
            return []
        if self.crop_class_id is None:
            #self.logger.error(f"[{self.__class__.__name__}] Cannot update: Crop Class ID not set.")
            return []

        start_time = time.monotonic()

        est_dx = motion_dx if motion_dx is not None else 0.0
        est_dy = motion_dy if motion_dy is not None else 0.0
        num_dets = len(detections)
        #self.logger.debug(f"[{self.__class__.__name__}] Update | Dets: {num_dets} | Motion: dx={est_dx:.1f}, dy={est_dy:.1f}")

        # Pre-computation: Extract detection properties 
        start_precomp_time = time.monotonic()
        det_centers = np.array([d.center for d in detections], dtype=np.float32) if num_dets > 0 else np.empty((0, 2), dtype=np.float32)
        det_areas = np.array([d.area for d in detections], dtype=np.float32) if num_dets > 0 else np.empty(0, dtype=np.float32)
        det_heights = np.array([d.height for d in detections], dtype=np.float32) if num_dets > 0 else np.empty(0, dtype=np.float32)
        det_aspects = np.array([d.width / max(1e-6, d.height) for d in detections], dtype=np.float32) if num_dets > 0 else np.empty(0, dtype=np.float32)
        det_is_near_edge = np.array([self.is_near_edge(d) for d in detections], dtype=bool) if num_dets > 0 else np.empty(0, dtype=bool)
        precomp_time = (time.monotonic() - start_precomp_time) * 1000

        # 1. Predict next state and prep track data 
        start_predict_time = time.monotonic()
        active_track_ids = list(self._tracks.keys())
        num_tracks = len(active_track_ids)
        track_predicted_pos = np.full((num_tracks, 2), np.nan, dtype=np.float32)
        track_last_areas = np.full(num_tracks, np.nan, dtype=np.float32)
        track_last_aspects = np.full(num_tracks, np.nan, dtype=np.float32)

        for i, track_id in enumerate(active_track_ids):
            track = self._tracks[track_id]
            track.frames_missing += 1
            last_pos = track.current_position 
            if last_pos:
                pred_x = last_pos[0] + est_dx
                pred_y = last_pos[1] + est_dy
                track.predicted_position = (pred_x, pred_y)
                track_predicted_pos[i] = [pred_x, pred_y]
                # Store last known area/aspect for cost calculation
                if track.current_detection:
                    track_last_areas[i] = track.current_detection.area
                    h = track.current_detection.height
                    track_last_aspects[i] = track.current_detection.width / max(1e-6, h)
            #else:
                #self.logger.warning(f"Track {track_id} has no last position.")
        predict_time = (time.monotonic() - start_predict_time) * 1000

        # 2. Calculate cost matrix with spacial pruning 
        start_cost_time = time.monotonic()
        cost_matrix = np.full((num_tracks, num_dets), INF_COST, dtype=np.float32)

        cost_calcs_done = 0
        for i, track_id in enumerate(active_track_ids):
            if np.isnan(track_predicted_pos[i, 0]):
                continue

            track = self._tracks[track_id]
            pred_x, pred_y = track_predicted_pos[i]

            search_x_min = pred_x - self._search_radius_x
            search_x_max = pred_x + self._search_radius_x
            search_y_min = pred_y - self._search_radius_y
            search_y_max = pred_y + self._search_radius_y

            # efficiently find candidate detections within search box 
            candidate_indices = np.where(
                (det_centers[:, 0] >= search_x_min) & (det_centers[:, 0] <= search_x_max) &
                (det_centers[:, 1] >= search_y_min) & (det_centers[:, 1] <= search_y_max)
            )[0]

            # calculate cost ONLY for candidates 
            for j in candidate_indices:
                detection = detections[j]
                # use pre-calculated track/detection properties
                cost, _ = self._compute_match_costs_optimized( 
                    track,
                    predicted_pos=(pred_x, pred_y),
                    last_area=track_last_areas[i],
                    last_aspect=track_last_aspects[i],
                    detection=detection,
                    det_center=det_centers[j],
                    det_area=det_areas[j],
                    det_aspect=det_aspects[j],
                    det_is_edge=det_is_near_edge[j]
                )
                cost_matrix[i, j] = cost
                cost_calcs_done += 1

        cost_calc_time = (time.monotonic() - start_cost_time) * 1000
        #self.logger.debug(f"Cost matrix: {num_tracks}x{num_dets}, Pruned Calcs: {cost_calcs_done} ({cost_calc_time:.1f} ms)")

        # 3. Assignment problem (Hungarian Algorithm) 
        start_assign_time = time.monotonic()
        matched_indices_list: List[Tuple[int, int]] = []
        if num_tracks > 0 and num_dets > 0 and cost_calcs_done > 0:
            try:
                # row_ind maps to track index, col_ind maps to detection index
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Filter out assignments with infinite cost (pruned/invalid matches)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < INF_COST:
                        matched_indices_list.append((r, c))

            except ValueError as e:
                 self.logger.error(f"Error during assignment: {e}. Cost matrix likely contains NaN/Inf improperly.")
            except Exception as e:
                 self.logger.error(f"Unexpected error during assignment: {e}\n{traceback.format_exc()}")
        assign_time = (time.monotonic() - start_assign_time) * 1000
        #self.logger.debug(f"Assignment found {len(matched_indices_list)} matches ({assign_time:.1f} ms)")

        # 4. Update matched tracks 
        start_update_time = time.monotonic() 
        matched_track_indices: Set[int] = set() 
        matched_det_indices: Set[int] = set() 

        for r, c in matched_indices_list:
            track_matrix_idx = r
            det_list_idx = c
            track_id = active_track_ids[track_matrix_idx]
            detection = detections[det_list_idx]
            track = self._tracks[track_id]

            # Update track state
            track.frames_missing = 0
            if track.current_detection:
                prev_pos = track.current_detection.center
                new_pos = detection.center
                track.last_horizontal_displacement = new_pos[0] - prev_pos[0]
                track.last_vertical_displacement = new_pos[1] - prev_pos[1]
            track.detections.append(detection)
            detection.track_id = track_id 

            # Update class history (using precomputed edge flag)
            is_edge = det_is_near_edge[det_list_idx]
            det_class_conf = detection.class_confidences if detection.class_confidences else {detection.class_id: detection.confidence}
            for class_id, conf in det_class_conf.items():
                if class_id not in track.class_history: track.class_history[class_id] = []
                track.class_history[class_id].append((conf, is_edge))

            # Handle crop locking
            if self._lock_crops_with_psez and not track.is_locked_to_crop and \
               detection.class_id == self.crop_class_id and detection.associated_psez:
                track.is_locked_to_crop = True
                #self.logger.info(f"Track {track.track_id} locked to CROP due to PSEZ.")

            matched_track_indices.add(track_matrix_idx)
            matched_det_indices.add(det_list_idx)

        # 5. Handle unmatched tracks / remove stale 
        stale_track_ids = []
        for i, track_id in enumerate(active_track_ids):
            if i not in matched_track_indices:
                track = self._tracks[track_id]
                if track.frames_missing > self._max_age:
                    stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            #self.logger.info(f"Removing stale Track {track_id} (missed {self._tracks[track_id].frames_missing} frames).")
            if self.finalize_callback:
                 try: 
                     self.finalize_callback(self._tracks[track_id])
                 except Exception as final_e: 
                     self.logger.error(f"Error finalizing track {track_id}: {final_e}")
            del self._tracks[track_id]

        # 6. Handle unmatched detections (create new tracks) 
        for j, detection in enumerate(detections):
            if j not in matched_det_indices:
                if detection.confidence >= self._min_confidence_new_track:
                    new_track_id = self._next_track_id
                    self._next_track_id += 1
                    detection.track_id = new_track_id 
                    new_track = TrackedObject(track_id=new_track_id, detections=[detection])

                    is_edge = det_is_near_edge[j]
                    det_class_conf = detection.class_confidences if detection.class_confidences else {detection.class_id: detection.confidence}
                    for class_id, conf in det_class_conf.items():
                         new_track.class_history[class_id] = [(conf, is_edge)]

                    # Handle initial crop locking
                    if self._lock_crops_with_psez and detection.class_id == self.crop_class_id and detection.associated_psez:
                         new_track.is_locked_to_crop = True

                    self._tracks[new_track_id] = new_track
                    #self.logger.debug(f"Created New Track {new_track_id} from Det {j} (Cls: {detection.class_id}, Conf: {detection.confidence:.2f})")

        update_time = (time.monotonic() - start_update_time) * 1000

        # 7. Return current active tracks 
        active_tracks = list(self._tracks.values())
        total_time_ms = (time.monotonic() - start_time) * 1000
        """
        self.logger.info(
            f"[{self.__class__.__name__}] Update finished: {total_time_ms:.1f} ms | "
            f"Precomp: {precomp_time:.1f} | Predict: {predict_time:.1f} | Cost: {cost_calc_time:.1f} ({cost_calcs_done} calcs) | "
            f"Assign: {assign_time:.1f} ({len(matched_indices_list)} matches) | Update: {update_time:.1f} | "
            f"Active Tracks: {len(active_tracks)}"
        )
        """
        return active_tracks

    def _compute_match_costs_optimized(self,
                                       track: TrackedObject,
                                       predicted_pos: Tuple[float, float],
                                       last_area: Optional[float],
                                       last_aspect: Optional[float],
                                       detection: Detection,
                                       det_center: np.ndarray, # Shape (2,)
                                       det_area: float,
                                       det_aspect: float,
                                       det_is_edge: bool
                                       ) -> Tuple[float, Dict[str, float]]: 
        detailed_costs = {} # Keep detailed costs optional for performance

        pred_x, pred_y = predicted_pos
        det_x, det_y = det_center

        # Position cost (normalized squared euclidean distance + penalty) 
        vert_diff = abs(det_y - pred_y)
        horiz_diff = abs(det_x - pred_x)

        # Calculate penalty factors only if exceeding base tolerance
        vert_penalty = max(1.0, vert_diff / (self._vert_tolerance_px * self._distance_penalty_threshold_factor)) if self._vert_tolerance_px > 0 else 1.0
        horiz_penalty = max(1.0, horiz_diff / (self._horiz_tolerance_px * self._distance_penalty_threshold_factor)) if self._horiz_tolerance_px > 0 else 1.0

        # Calculate normalized squared differences
        norm_vert_sq_diff = (vert_diff / max(1e-6, self._vert_tolerance_px))**2 * vert_penalty
        norm_horiz_sq_diff = (horiz_diff / max(1e-6, self._horiz_tolerance_px))**2 * horiz_penalty

        position_cost = norm_vert_sq_diff # Primarily vertical focus as rows are straight
        horizontal_cost_term = norm_horiz_sq_diff

        # Area cost 
        area_cost = 0.5 # Default
        if last_area is not None and not np.isnan(last_area) and last_area > 1e-6 and det_area > 1e-6:
             max_a = max(det_area, last_area)
             min_a = min(det_area, last_area)
             area_ratio = min_a / max_a
             area_cost = 1.0 - area_ratio

        # Aspect ratio cost 
        aspect_cost = 0.5 # Default
        if last_aspect is not None and not np.isnan(last_aspect) and last_aspect > 1e-6 and det_aspect > 1e-6:
            max_asp = max(det_aspect, last_aspect)
            min_asp = min(det_aspect, last_aspect)
            aspect_ratio = min_asp / max_asp
            aspect_cost = 1.0 - aspect_ratio

        # Class consistency cost 
        class_consistency_cost = 0.0
        predicted_class_id, _ = self.get_predicted_class(track) # Still need this helper? Should be improved
        if track.is_locked_to_crop and detection.class_id != self.crop_class_id:
            class_consistency_cost = 1.0
        elif predicted_class_id != -1 and detection.class_id != predicted_class_id:
            class_consistency_cost = 0.75 # Penalty for class change

        # Edge penalty 
        edge_penalty = 0.0
        if det_is_edge:
            # Scale the dominant position cost component
            edge_penalty = self._pos_weight * (1.0 / max(0.1, self._bbox_touching_edge_weight) - 1.0)

        # Total weighted cost 
        total_cost = (
            self._pos_weight * position_cost +
            self._horiz_weight * horizontal_cost_term +
            self._area_weight * area_cost +
            self._aspect_weight * aspect_cost +
            self._class_weight * class_consistency_cost +
            edge_penalty
        )

        return max(0.0, total_cost), detailed_costs

    def is_near_edge(self, detection: Detection) -> bool:
        x_min, y_min, x_max, y_max = detection.bbox
        return (x_min < self._edge_threshold_px or
                x_max > self._image_width - self._edge_threshold_px or
                y_min < self._edge_threshold_px or
                y_max > self._image_height - self._edge_threshold_px)

    def get_predicted_class(self, track: TrackedObject) -> Tuple[int, float]:
        if not track.detections: return -1, 0.0

        # Handle locked state first
        if track.is_locked_to_crop:
            crop_confidences = []
            if self.crop_class_id in track.class_history:
                 crop_confidences = [conf for conf, is_edge in track.class_history[self.crop_class_id]]
            # Use the highest confidence seen for the crop, or fallback
            best_score = max(crop_confidences) if crop_confidences else track.current_detection.confidence if track.current_detection else 0.0
            return self.crop_class_id, best_score

        # Calculate weighted scores based on history 
        class_scores = defaultdict(float)
        class_weights = defaultdict(float)
        has_history = False
        for class_id, history_entries in track.class_history.items():
             if history_entries: has_history = True
             for conf, is_edge in history_entries:
                 weight = self._bbox_touching_edge_weight if is_edge else 1.0
                 class_scores[class_id] += conf * weight
                 class_weights[class_id] += weight

        if not has_history or not class_scores:
            # Fallback to the very last detection if no valid history
            current_det = track.current_detection
            return (current_det.class_id, current_det.confidence) if current_det else (-1, 0.0)

        # Normalize scores by total weight
        averaged_scores = {
            cid: class_scores[cid] / max(1e-6, class_weights[cid])
            for cid in class_scores
        }

        # Get the class with the highest average weighted score
        try:
            best_class, best_avg_score = max(averaged_scores.items(), key=lambda item: item[1])
        except ValueError:
             current_det = track.current_detection
             return (current_det.class_id, current_det.confidence) if current_det else (-1, 0.0)

        return best_class, best_avg_score