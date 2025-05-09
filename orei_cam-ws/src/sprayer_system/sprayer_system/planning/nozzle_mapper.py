import logging
from typing import List, Dict, Tuple, Optional
from ..common.definitions import ManagedObject, Nozzle
from .nozzle_configuration import NozzleConfiguration

class NozzleMapper:
    """
    Maps target objects to the nozzles whose spray patterns cover them laterally (X-axis)
    Uses a fast 1D interval overlap check based on precomputed absolute bounding boxes
    """
    def __init__(self, nozzle_config: NozzleConfiguration, logger: logging.Logger):
        self._nozzle_config = nozzle_config
        self._logger = logger
        self._all_nozzles = self._nozzle_config.get_all_nozzles() 
        self._nozzle_abs_x_intervals: List[Tuple[float, float]] = []

        self._precompute_nozzle_intervals()

        if not self._all_nozzles:
            self._logger.error("NozzleMapper initialized with zero nozzles!")
        elif len(self._nozzle_abs_x_intervals) != len(self._all_nozzles):
             self._logger.error("NozzleMapper init failed: Mismatch between nozzle count and precomputed absolute intervals!")
             self._nozzle_abs_x_intervals = []
        else:
             self._logger.info(f"NozzleMapper initialized with {len(self._all_nozzles)} nozzles and precomputed absolute X intervals.")

    def _precompute_nozzle_intervals(self):
        """Calculates the absolute ground X-interval for each nozzle's spray pattern"""
        self._logger.info("Precomputing nozzle absolute X-intervals...")
        self._nozzle_abs_x_intervals = []

        for nozzle in self._all_nozzles:
            abs_nozzle_x, _ = nozzle.position_m
            rel_bbox = nozzle.bounding_box_relative_m

            if rel_bbox:
                rel_min_x, _, rel_max_x, _ = rel_bbox
                abs_interval_min_x = abs_nozzle_x + rel_min_x
                abs_interval_max_x = abs_nozzle_x + rel_max_x
                self._nozzle_abs_x_intervals.append((abs_interval_min_x, abs_interval_max_x))
                # self._logger.debug(
                #    f"  Nozzle {nozzle.index}: PosX={abs_nozzle_x:.3f}, RelBBoxX=({rel_min_x:.3f}, {rel_max_x:.3f}) "
                #    f"-> Abs Interval=({abs_interval_min_x:.3f}, {abs_interval_max_x:.3f})"
                #)
            else:
                self._logger.warning(f"Nozzle {nozzle.index} has no relative bounding box. Using degenerate absolute interval at {abs_nozzle_x:.3f}m.")
                self._nozzle_abs_x_intervals.append((abs_nozzle_x, abs_nozzle_x))

        if len(self._nozzle_abs_x_intervals) != len(self._all_nozzles):
             self._logger.error(f"Precomputation mismatch: Expected {len(self._all_nozzles)} intervals, got {len(self._nozzle_abs_x_intervals)}.")


    def map_objects_to_nozzles(self, objects: List[ManagedObject]) -> Dict[int, List[int]]:
        """
        Maps objects to nozzles based on lateral (X-axis) overlap between object's predicted bounding box and nozzle's absolute spray bounding box
        """
        mapping: Dict[int, List[int]] = {}
        num_nozzles = len(self._nozzle_abs_x_intervals)

        if num_nozzles == 0:
            # self._logger.debug("No nozzle intervals precomputed, cannot perform mapping.")
            return mapping 

        num_input_objects = len(objects)
        num_mapped = 0

        for obj in objects:
            # Ensure the object's absolute bounding box is calculated
            # ObjectManager should ideally call this after prediction update
            if obj.bounding_box_m is None:
                obj.update_bounding_box_m()

            if obj.bounding_box_m is None:
                # self._logger.debug(f"Object {obj.track_id} has no absolute bounding box. Skipping mapping.")
                continue

            try:
                obj_min_x, _, obj_max_x, _ = obj.bounding_box_m

                # basic sanity check for object bbox validity
                if obj_min_x >= obj_max_x:
                    # self._logger.debug(f"Object {obj.track_id} has invalid X bounds ({obj_min_x:.3f}, {obj_max_x:.3f}). Skipping mapping.")
                    continue

                covering_nozzles: List[int] = []

                for nozzle_index in range(num_nozzles):
                    # Use the precomputed ABSOLUTE nozzle interval
                    nozzle_abs_min_x, nozzle_abs_max_x = self._nozzle_abs_x_intervals[nozzle_index]

                    # Fast 1D interval overlap check
                    # Checks if the object interval [obj_min_x, obj_max_x] overlaps with the nozzle interval [nozzle_abs_min_x, nozzle_abs_max_x]
                    overlap = (obj_min_x < nozzle_abs_max_x) and (obj_max_x > nozzle_abs_min_x)
                    # --------------------------------------

                    if overlap:
                        covering_nozzles.append(nozzle_index)
                    # If there's no overlap AND this nozzle starts AFTER the object ends, then no nozzles further right can overlap either (assuming nozzles ordered L->R)
                    elif nozzle_abs_min_x >= obj_max_x:
                        break
                    # ----------------------------

                if covering_nozzles:
                    mapping[obj.track_id] = covering_nozzles
                    num_mapped += 1
                    # self._logger.debug(f"Mapped Target {obj.track_id} (X: {obj_min_x:.2f}-{obj_max_x:.2f}) to Nozzles: {covering_nozzles}")

            except IndexError:
                 self._logger.error(f"IndexError during mapping for Obj {obj.track_id}. Nozzle intervals/indices mismatch?")
                 break
            except Exception as e:
                 self._logger.error(f"Unexpected error mapping object {obj.track_id}: {e}")

        # self._logger.info(f"Mapped {num_mapped}/{num_input_objects} objects to nozzles.")
        return mapping