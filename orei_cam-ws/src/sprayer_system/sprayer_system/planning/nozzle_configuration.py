# /mnt/c/Users/Mic45/Documents/AU/jetson_simulation/orei-cam_ws/src/sprayer_system/sprayer_system/planning/nozzle_configuration.py

import logging
import os
import yaml
from typing import List, Dict, Any, Tuple, Optional
import numpy as np # For min/max calculations on pattern pixels

from ..common.definitions import Nozzle, CameraParams
from ..common import geometry

class NozzleConfiguration:
    """
    - Loads, validates, and manages the nozzle setup for the spray boom
    - Can load from a calibration YAML file or generate a default layout
    - Handles scaling of calibration pixel coordinates if calibration image dimensions differ from the node's configured dimensions
    - Crucially, it now calculates an 'activation_y_m' for each nozzle, representing the Y-center of its spray pattern on the ground
    - Allows shrinking of loaded calibration polygons to a target width/height
    """

    def __init__(self,
                 layout_config: Dict[str, Any],
                 node_cam_params: CameraParams,
                 logger: logging.Logger):
        self._config = layout_config
        self._cam_params = node_cam_params
        self._logger = logger
        self._nozzles: List[Nozzle] = []

        self._num_nozzles_config = self._config.get('num_nozzles', 0)
        if not isinstance(self._num_nozzles_config, int) or self._num_nozzles_config <= 0:
            raise ValueError("NozzleConfiguration: 'num_nozzles' must be a positive integer in config.")

        self._activation_zone_target_width_px = float(self._config.get('activation_zone_target_width_px', -1.0))
        self._activation_zone_target_height_px = float(self._config.get('activation_zone_target_height_px', -1.0))
        
        if self._activation_zone_target_width_px > 0 or self._activation_zone_target_height_px > 0:
            self._logger.info(
                f"Nozzle activation zone shrinkage configured: "
                f"Target Width={self._activation_zone_target_width_px if self._activation_zone_target_width_px > 0 else 'N/A'} px, "
                f"Target Height={self._activation_zone_target_height_px if self._activation_zone_target_height_px > 0 else 'N/A'} px. "
                f"(Applies to loaded calibration polygons)"
            )

        self._load_configuration()

    def _load_configuration(self):
        calib_file_path = self._config.get('calibration_file', None)
        use_placeholder_fallback = self._config.get('placeholder', True)
        loaded_from_calib = False

        if calib_file_path and os.path.exists(calib_file_path):
            try:
                self._logger.info(f"Attempting to load nozzle layout from calibration file: {calib_file_path}")
                self._nozzles = self._load_from_calibration_file(calib_file_path)
                loaded_from_calib = True
                self._logger.info(f"Successfully loaded layout for {len(self._nozzles)} nozzles from calibration.")
            except Exception as e:
                self._logger.error(f"Failed to load from calibration file '{calib_file_path}': {e}.", exc_info=True)
                self._nozzles = []
                loaded_from_calib = False
        elif calib_file_path:
            self._logger.warning(f"Calibration file specified but not found: {calib_file_path}")

        if not loaded_from_calib:
            if use_placeholder_fallback:
                self._logger.info("Falling back to default nozzle layout generation.")
                try:
                    self._nozzles = self._generate_default_layout(self._num_nozzles_config)
                    self._logger.info(f"Generated default layout for {len(self._nozzles)} nozzles.")
                except Exception as e:
                    self._logger.error(f"Failed to generate default layout: {e}", exc_info=True)
                    self._nozzles = []
            else:
                self._logger.error("Calibration file invalid/not found AND placeholder fallback is disabled. No nozzles loaded.")
                self._nozzles = []

        if len(self._nozzles) != self._num_nozzles_config:
            self._logger.warning(
                f"Final nozzle count ({len(self._nozzles)}) does not match configured 'num_nozzles' ({self._num_nozzles_config}). "
                "This may occur if calibration file has fewer entries or default generation was capped."
            )
        elif not self._nozzles:
            self._logger.error("CRITICAL: No nozzles were successfully loaded or generated.")
        else:
            self._logger.info(f"Nozzle configuration complete. Final count: {len(self._nozzles)}")
            for nzl in self._nozzles:
                 if nzl.activation_y_m is not None:
                      self._logger.debug(f"  Nozzle {nzl.index}: Activation Y = {nzl.activation_y_m:.3f} m")
                 else:
                      self._logger.warning(f"  Nozzle {nzl.index}: Activation Y is None!")


    def _load_from_calibration_file(self, file_path: str) -> List[Nozzle]:
        try:
            with open(file_path, 'r') as f:
                calib_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file not found at {file_path}") from None
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML calibration file {file_path}: {e}") from e
        except OSError as e:
            raise OSError(f"OS error reading calibration file {file_path}: {e}") from e

        if not isinstance(calib_data, dict):
            raise ValueError("Calibration file content is not a dictionary.")

        calib_details = calib_data.get('calibration_details', {})
        calib_nozzles_data = calib_data.get('nozzles', [])

        calib_img_w = calib_details.get('image_width_px')
        calib_img_h = calib_details.get('image_height_px')
        if not isinstance(calib_img_w, (int, float)) or calib_img_w <= 0 or \
           not isinstance(calib_img_h, (int, float)) or calib_img_h <= 0:
            raise ValueError("Invalid or missing 'image_width_px' or 'image_height_px' in calibration_details.")

        node_w = self._cam_params.width_px
        node_h = self._cam_params.height_px
        calib_center_x_img = float(calib_img_w) / 2.0
        calib_center_y_img = float(calib_img_h) / 2.0
        node_center_x_img = self._cam_params.center_x_px
        node_center_y_img = self._cam_params.center_y_px

        scale_x = node_w / calib_img_w if calib_img_w > 0 else 1.0
        scale_y = node_h / calib_img_h if calib_img_h > 0 else 1.0
        needs_scaling = (abs(calib_img_w - node_w) > 1e-3 or abs(calib_img_h - node_h) > 1e-3)

        if needs_scaling:
            self._logger.info(
                f"Scaling calibration pixels: Calib ({calib_img_w}x{calib_img_h}) -> Node ({node_w}x{node_h}). "
                f"Scale X={scale_x:.4f}, Y={scale_y:.4f}"
            )
        else:
            self._logger.info("Calibration image dimensions match node configuration. No scaling needed for patterns.")

        try:
            base_nozzles_for_pos = self._generate_default_layout(self._num_nozzles_config)
            base_nozzle_map = {n.index: n for n in base_nozzles_for_pos}
        except Exception as e:
            self._logger.error(f"Failed to generate base nozzle positions for calibration loading: {e}")
            raise RuntimeError("Cannot load calibration without base nozzle positions.") from e

        processed_nozzles_dict: Dict[int, Nozzle] = {}

        for nozzle_data_calib in calib_nozzles_data:
            if not isinstance(nozzle_data_calib, dict): continue

            index = nozzle_data_calib.get('index')
            pattern_px_calib_raw = nozzle_data_calib.get('spray_pattern_pixels')

            if index is None or not isinstance(index, int) or index < 0 or index >= self._num_nozzles_config:
                self._logger.warning(f"Skipping nozzle entry with invalid/out-of-range index: {index} from {nozzle_data_calib}")
                continue
            if index in processed_nozzles_dict:
                 self._logger.warning(f"Duplicate index {index} found in calibration. Skipping subsequent entry.")
                 continue
            if not isinstance(pattern_px_calib_raw, list) or not pattern_px_calib_raw:
                self._logger.warning(f"Skipping nozzle index {index} due to missing/empty 'spray_pattern_pixels'.")
                continue
            
            if index in base_nozzle_map:
                current_nozzle = base_nozzle_map[index]
            else:
                self._logger.error(f"Internal error: Index {index} valid but not in base_nozzle_map.")
                continue

            pattern_px_calib_np = np.array(pattern_px_calib_raw, dtype=np.float32)
            if pattern_px_calib_np.ndim != 2 or pattern_px_calib_np.shape[1] != 2 or pattern_px_calib_np.shape[0] == 0:
                self._logger.warning(f"Nozzle {index} has invalid spray_pattern_pixels shape: {pattern_px_calib_np.shape}. Skipping.")
                continue

            # apply Activation Zone Shrinking (if configured)
            if self._activation_zone_target_width_px > 0 or self._activation_zone_target_height_px > 0:
                if pattern_px_calib_np.shape[0] >= 3: # Need at least 3 points for a valid area
                    try:
                        orig_min_x_calib = np.min(pattern_px_calib_np[:, 0])
                        orig_max_x_calib = np.max(pattern_px_calib_np[:, 0])
                        orig_min_y_calib = np.min(pattern_px_calib_np[:, 1])
                        orig_max_y_calib = np.max(pattern_px_calib_np[:, 1])

                        orig_width_calib = orig_max_x_calib - orig_min_x_calib
                        orig_height_calib = orig_max_y_calib - orig_min_y_calib
                        
                        center_x_calib = (orig_min_x_calib + orig_max_x_calib) / 2.0
                        center_y_calib = (orig_min_y_calib + orig_max_y_calib) / 2.0

                        new_poly_min_x_calib = orig_min_x_calib
                        new_poly_max_x_calib = orig_max_x_calib
                        new_poly_min_y_calib = orig_min_y_calib
                        new_poly_max_y_calib = orig_max_y_calib
                        
                        width_changed = False
                        height_changed = False

                        if self._activation_zone_target_width_px > 0 and orig_width_calib > 1e-3:
                            target_w = self._activation_zone_target_width_px
                            final_w = min(orig_width_calib, target_w)
                            if abs(final_w - orig_width_calib) > 1e-3 :
                                new_poly_min_x_calib = center_x_calib - final_w / 2.0
                                new_poly_max_x_calib = center_x_calib + final_w / 2.0
                                width_changed = True
                        
                        if self._activation_zone_target_height_px > 0 and orig_height_calib > 1e-3:
                            target_h = self._activation_zone_target_height_px
                            final_h = min(orig_height_calib, target_h)
                            if abs(final_h - orig_height_calib) > 1e-3:
                                new_poly_min_y_calib = center_y_calib - final_h / 2.0
                                new_poly_max_y_calib = center_y_calib + final_h / 2.0
                                height_changed = True

                        if width_changed or height_changed:
                            pattern_px_calib_np = np.array([
                                [new_poly_min_x_calib, new_poly_min_y_calib], # Top-left
                                [new_poly_max_x_calib, new_poly_min_y_calib], # Top-right
                                [new_poly_max_x_calib, new_poly_max_y_calib], # Bottom-right
                                [new_poly_min_x_calib, new_poly_max_y_calib]  # Bottom-left
                            ], dtype=np.float32)
                            self._logger.info(
                                f"  Nozzle {index}: Activation zone polygon shrunk. "
                                f"New calib_px bounds: X=[{new_poly_min_x_calib:.1f}, {new_poly_max_x_calib:.1f}], "
                                f"Y=[{new_poly_min_y_calib:.1f}, {new_poly_max_y_calib:.1f}]"
                            )
                    except Exception as shrink_e:
                        self._logger.error(f"Error during activation zone shrinking for nozzle {index}: {shrink_e}. Using original polygon.")

            pattern_relative_m: List[Tuple[float, float]] = []
            valid_pattern_conversion = True
            for point_px_calib in pattern_px_calib_np: # iterate (maybe shrunk) polygon points
                try:
                    calib_x, calib_y = point_px_calib
                    equiv_node_px_x, equiv_node_px_y = calib_x, calib_y
                    if needs_scaling:
                        rel_calib_x = calib_x - calib_center_x_img
                        rel_calib_y = calib_y - calib_center_y_img
                        scaled_rel_x = rel_calib_x * scale_x
                        scaled_rel_y = rel_calib_y * scale_y
                        equiv_node_px_x = scaled_rel_x + node_center_x_img
                        equiv_node_px_y = scaled_rel_y + node_center_y_img

                    abs_ground_m = geometry.image_px_to_robot_ground_m(
                        (equiv_node_px_x, equiv_node_px_y), self._cam_params
                    )
                    rel_m_x = abs_ground_m[0] - current_nozzle.position_m[0]
                    rel_m_y = abs_ground_m[1] - current_nozzle.position_m[1]
                    pattern_relative_m.append((rel_m_x, rel_m_y))
                except Exception as conv_e:
                    self._logger.error(f"Error converting/scaling point {point_px_calib} for nozzle {index}: {conv_e}")
                    valid_pattern_conversion = False; break
            
            if not valid_pattern_conversion:
                self._logger.warning(f"Failed to convert full pattern for nozzle {index}. It will retain default pattern if any.")
                processed_nozzles_dict[index] = current_nozzle
                continue

            current_nozzle.spray_pattern_relative_m = pattern_relative_m
            current_nozzle.recalculate_bounding_box()

            calib_pat_min_x = np.min(pattern_px_calib_np[:, 0])
            calib_pat_max_x = np.max(pattern_px_calib_np[:, 0])
            calib_pat_min_y = np.min(pattern_px_calib_np[:, 1])
            calib_pat_max_y = np.max(pattern_px_calib_np[:, 1])

            calib_pat_center_x_px = (calib_pat_min_x + calib_pat_max_x) / 2.0
            calib_pat_center_y_px = (calib_pat_min_y + calib_pat_max_y) / 2.0

            equiv_node_pat_center_x_px, equiv_node_pat_center_y_px = calib_pat_center_x_px, calib_pat_center_y_px
            if needs_scaling:
                rel_calib_ctr_x = calib_pat_center_x_px - calib_center_x_img
                rel_calib_ctr_y = calib_pat_center_y_px - calib_center_y_img
                scaled_rel_ctr_x = rel_calib_ctr_x * scale_x
                scaled_rel_ctr_y = rel_calib_ctr_y * scale_y
                equiv_node_pat_center_x_px = scaled_rel_ctr_x + node_center_x_img
                equiv_node_pat_center_y_px = scaled_rel_ctr_y + node_center_y_img
            
            try:
                abs_ground_pattern_center_m = geometry.image_px_to_robot_ground_m(
                    (equiv_node_pat_center_x_px, equiv_node_pat_center_y_px), self._cam_params
                )
                current_nozzle.activation_y_m = abs_ground_pattern_center_m[1]
            except Exception as e:
                self._logger.error(f"Error converting pattern center to ground for nozzle {index} activation_y_m: {e}. Using default.")

            processed_nozzles_dict[index] = current_nozzle
            
        final_nozzles_list: List[Nozzle] = []
        for i in range(self._num_nozzles_config):
            if i in processed_nozzles_dict:
                final_nozzles_list.append(processed_nozzles_dict[i])
            elif i in base_nozzle_map: 
                self._logger.warning(f"Nozzle {i} not in calibration file. Using default pattern and activation Y.")
                final_nozzles_list.append(base_nozzle_map[i])

        final_nozzles_list.sort(key=lambda n: n.index)
        return final_nozzles_list


    def _generate_default_layout(self, num_nozzles: int) -> List[Nozzle]:
        try:
            spacing_cm = float(self._config['default_spacing_cm'])
            spray_width_cm = float(self._config['default_spray_width_cm'])
            spray_length_cm = float(self._config['default_spray_length_cm'])
            default_nozzle_center_y_m = float(self._config.get('default_boom_y_position_m', 0.0))
        except KeyError as e:
            raise KeyError(f"Missing required default layout parameter: '{e}'") from e
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numerical value for default layout parameter: {e}") from e

        if spacing_cm <= 0: raise ValueError("default_spacing_cm must be positive")
        if spray_width_cm <= 0: raise ValueError("default_spray_width_cm must be positive")
        if spray_length_cm <= 0: raise ValueError("default_spray_length_cm must be positive")

        spacing_m = spacing_cm / 100.0
        spray_width_m = spray_width_cm / 100.0
        spray_length_m = spray_length_cm / 100.0

        self._logger.info(f"  Generating Default Layout Params: Spacing={spacing_cm}cm, "
                          f"Spray W={spray_width_cm}cm, Spray L={spray_length_cm}cm, "
                          f"Default Nozzle Y Center={default_nozzle_center_y_m:.3f}m")

        nozzles_list: List[Nozzle] = []
        total_boom_width = (num_nozzles - 1) * spacing_m if num_nozzles > 0 else 0
        start_x = -total_boom_width / 2.0

        half_width = spray_width_m / 2.0
        half_length = spray_length_m / 2.0
        default_pattern_relative_m = [
            (-half_width, -half_length), ( half_width, -half_length),
            ( half_width,  half_length), (-half_width,  half_length)
        ]

        for i in range(num_nozzles):
            pos_x = start_x + i * spacing_m
            nozzle = Nozzle(
                index=i,
                position_m=(pos_x, default_nozzle_center_y_m), 
                spray_pattern_relative_m=list(default_pattern_relative_m)
            )
            nozzle.activation_y_m = default_nozzle_center_y_m
            nozzles_list.append(nozzle)

        return nozzles_list

    def get_nozzle(self, index: int) -> Optional[Nozzle]:
        for nozzle in self._nozzles:
            if nozzle.index == index:
                return nozzle
        self._logger.warning(f"Attempted to get invalid nozzle index: {index}")
        return None

    def get_all_nozzles(self) -> List[Nozzle]:
        return list(self._nozzles) 

    @property
    def num_nozzles(self) -> int:
        return len(self._nozzles)