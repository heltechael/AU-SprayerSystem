import argparse
import os
import sys
import datetime
import logging
from typing import List, Dict, Tuple, Optional, Any

try:
    import yaml
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}. Please install PyYAML and NumPy.")
    print("pip install PyYAML numpy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCALE_FACTOR = 2

class NozzleData:
    def __init__(self, index: int, polygon_px: List[List[int]]):
        self.index = index
        self.original_polygon_px = polygon_px
        self.min_x: float = float('inf')
        self.max_x: float = float('-inf')
        self.min_y: float = float('inf')
        self.max_y: float = float('-inf')
        self.center_x: Optional[float] = None
        self.pruned_min_x: Optional[float] = None
        self.pruned_max_x: Optional[float] = None
        self._calculate_bounds_and_center()

    def _calculate_bounds_and_center(self):
        if not self.original_polygon_px:
            logger.warning(f"Nozzle {self.index} has no polygon points.")
            return

        points = np.array(self.original_polygon_px, dtype=np.float32)
        if points.shape[0] < 1 or points.shape[1] != 2:
             logger.warning(f"Nozzle {self.index} has invalid polygon shape: {points.shape}")
             return

        self.min_x = float(np.min(points[:, 0]))
        self.max_x = float(np.max(points[:, 0]))
        self.min_y = float(np.min(points[:, 1]))
        self.max_y = float(np.max(points[:, 1]))

        if self.min_x <= self.max_x:
            self.center_x = (self.min_x + self.max_x) / 2.0
        else:
             logger.warning(f"Nozzle {self.index} has min_x ({self.min_x}) > max_x ({self.max_x}).")


def prune_nozzles(input_file: str, output_file: str):
    logger.info(f"Loading nozzle calibration from: {input_file}")

    try:
        with open(input_file, 'r') as f:
            calib_data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input file {input_file}: {e}")
        sys.exit(1)

    if not isinstance(calib_data, dict) or 'nozzles' not in calib_data:
        logger.error("Invalid YAML structure: Missing 'nozzles' key.")
        sys.exit(1)

    nozzles_raw = calib_data.get('nozzles', [])
    nozzle_objects: List[NozzleData] = []
    valid_indices = set()
    for nozzle_dict in nozzles_raw:
        if not isinstance(nozzle_dict, dict): continue
        idx = nozzle_dict.get('index')
        poly = nozzle_dict.get('spray_pattern_pixels')
        if isinstance(idx, int) and idx >= 0 and isinstance(poly, list) and poly:
            if idx in valid_indices:
                 logger.warning(f"Duplicate index {idx} found. Skipping subsequent entry.")
                 continue
            try:
                nozzle_obj = NozzleData(idx, poly)
                if nozzle_obj.center_x is not None: 
                    nozzle_objects.append(nozzle_obj)
                    valid_indices.add(idx)
                else:
                     logger.warning(f"Skipping nozzle {idx} due to invalid polygon data.")
            except Exception as e:
                 logger.error(f"Error processing nozzle {idx} data: {e}")
        else:
            logger.warning(f"Skipping invalid nozzle entry: {nozzle_dict}")

    if not nozzle_objects:
        logger.error("No valid nozzle data found in the input file.")
        sys.exit(1)

    nozzle_objects.sort(key=lambda n: n.index)
    nozzle_map: Dict[int, NozzleData] = {n.index: n for n in nozzle_objects}
    sorted_indices = [n.index for n in nozzle_objects]
    num_valid_nozzles = len(sorted_indices)
    logger.info(f"Processed {num_valid_nozzles} valid nozzles. Indices: {sorted_indices}")

    logger.info("Calculating pruned boundaries based on midpoints...")
    for i, current_idx in enumerate(sorted_indices):
        current_nozzle = nozzle_map[current_idx]

        if i == 0: 
            current_nozzle.pruned_min_x = current_nozzle.min_x
            logger.debug(f"Nozzle {current_idx} (Leftmost): Using original min_x: {current_nozzle.pruned_min_x:.2f}")
        else:
            left_neighbor_idx = sorted_indices[i-1]
            left_neighbor = nozzle_map[left_neighbor_idx]
            if left_neighbor.center_x is not None and current_nozzle.center_x is not None:
                 midpoint_x = (left_neighbor.center_x + current_nozzle.center_x) / 2.0
                 current_nozzle.pruned_min_x = midpoint_x
                 logger.debug(f"Nozzle {current_idx}: Left boundary (midpoint with {left_neighbor_idx}): {current_nozzle.pruned_min_x:.2f}")
            else:
                 logger.warning(f"Cannot calculate left boundary for {current_idx} due to missing center_x on neighbor {left_neighbor_idx}. Using original min_x.")
                 current_nozzle.pruned_min_x = current_nozzle.min_x

        if i == num_valid_nozzles - 1: 
            current_nozzle.pruned_max_x = current_nozzle.max_x
            logger.debug(f"Nozzle {current_idx} (Rightmost): Using original max_x: {current_nozzle.pruned_max_x:.2f}")
        else:
            right_neighbor_idx = sorted_indices[i+1]
            right_neighbor = nozzle_map[right_neighbor_idx]
            if right_neighbor.center_x is not None and current_nozzle.center_x is not None:
                 midpoint_x = (current_nozzle.center_x + right_neighbor.center_x) / 2.0
                 current_nozzle.pruned_max_x = midpoint_x
                 logger.debug(f"Nozzle {current_idx}: Right boundary (midpoint with {right_neighbor_idx}): {current_nozzle.pruned_max_x:.2f}")
            else:
                 logger.warning(f"Cannot calculate right boundary for {current_idx} due to missing center_x on neighbor {right_neighbor_idx}. Using original max_x.")
                 current_nozzle.pruned_max_x = current_nozzle.max_x

        if current_nozzle.pruned_min_x is not None and current_nozzle.pruned_max_x is not None:
             if current_nozzle.pruned_min_x >= current_nozzle.pruned_max_x:
                  logger.warning(f"Nozzle {current_idx}: Pruned min_x ({current_nozzle.pruned_min_x:.2f}) >= pruned max_x ({current_nozzle.pruned_max_x:.2f}). Check nozzle spacing/overlap.")

    output_data: Dict[str, Any] = {
        'calibration_details': calib_data.get('calibration_details', {}), 
        'nozzles': []
    }
    output_data['calibration_details']['pruning_timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_data['calibration_details']['pruning_input_file'] = os.path.basename(input_file)

    num_pruned = 0
    for nozzle_idx in sorted_indices: 
        nozzle = nozzle_map[nozzle_idx]
        if nozzle.pruned_min_x is not None and nozzle.pruned_max_x is not None:
            pruned_poly_px = [
                [nozzle.pruned_min_x*SCALE_FACTOR, nozzle.min_y*SCALE_FACTOR], # Top-left
                [nozzle.pruned_max_x*SCALE_FACTOR, nozzle.min_y*SCALE_FACTOR], # Top-right
                [nozzle.pruned_max_x*SCALE_FACTOR, nozzle.max_y*SCALE_FACTOR], # Bottom-right
                [nozzle.pruned_min_x*SCALE_FACTOR, nozzle.max_y*SCALE_FACTOR], # Bottom-left
            ]
            output_data['nozzles'].append({
                'index': nozzle.index,
                'spray_pattern_pixels': pruned_poly_px 
            })
            num_pruned += 1
        else:
            logger.warning(f"Skipping nozzle {nozzle_idx} in output due to failed pruning calculation.")

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(output_data, f, indent=2, sort_keys=False, default_flow_style=None)
        logger.info(f"Successfully saved pruned calibration data for {num_pruned} nozzles to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to save pruned calibration data to {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune nozzle calibration polygons to remove overlap.")
    parser.add_argument(
        "input_file",
        help="Path to the input nozzle_calibration.yaml file."
    )
    parser.add_argument(
        "-o", "--output-file",
        default="pruned_nozzle_calibration.yaml",
        help="Path to save the pruned calibration YAML file (default: pruned_nozzle_calibration.yaml)."
    )

    args = parser.parse_args()

    prune_nozzles(args.input_file, args.output_file)