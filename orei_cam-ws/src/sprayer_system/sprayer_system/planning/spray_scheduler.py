import logging
import math 
from typing import List, Dict, Any, Tuple, Optional

from rclpy.time import Time, Duration 
from rclpy.clock import Clock 

from ..common.definitions import ManagedObject, ObjectStatus, Nozzle
from .nozzle_configuration import NozzleConfiguration

SERIAL_SEND_DELAY_SEC = 0.001 
RELAY_RESPONSE_DELAY_SEC = 0.001 
NOZZLE_ACTUATION_ON_DELAY_SEC = 0.005 

class SprayScheduler:
    """
    Determines the current nozzle activation state.
    1. Activates nozzles based on TARGETED objects' future activation_bounding_box overlap.
    2. Deactivates nozzles if they would spray into a protected object's future safety_bounding_box.
    Considers latency and spray margins.
    """
    def __init__(self,
                 timing_config: Dict[str, Any],
                 nozzle_config: NozzleConfiguration,
                 logger: logging.Logger,
                 clock: Clock): 
        self._config = timing_config
        self._nozzle_config = nozzle_config
        self._logger = logger
        
        try:
            self._nozzle_actuation_latency_sec = float(self._config.get('nozzle_actuation_latency', 0.050))
            self._spray_margin_sec = float(self._config.get('spray_margin_time', 0.020))

            self._min_physical_system_latency_sec = (
                SERIAL_SEND_DELAY_SEC +
                RELAY_RESPONSE_DELAY_SEC +
                NOZZLE_ACTUATION_ON_DELAY_SEC
            )
            
            self._effective_latency_sec = self._nozzle_actuation_latency_sec
            if self._nozzle_actuation_latency_sec < self._min_physical_system_latency_sec:
                 self._logger.warning(
                     f"Configured 'nozzle_actuation_latency' ({self._nozzle_actuation_latency_sec:.3f}s) "
                     f"is less than estimated minimum physical system delays ({self._min_physical_system_latency_sec:.3f}s). "
                     "Timing might be optimistic."
                 )
            
            if self._spray_margin_sec < 0:
                self._logger.warning(f"spray_margin_time ({self._spray_margin_sec:.3f}s) is negative. Clamping to 0.0s.")
                self._spray_margin_sec = 0.0

            self._logger.info("SprayScheduler Initialized:")
            self._logger.info(f"  Effective Actuation Latency for Prediction: {self._effective_latency_sec:.3f}s")
            self._logger.info(f"  Spray Margin (temporal padding for targets): {self._spray_margin_sec:.3f}s")

        except (ValueError, TypeError) as e:
            self._logger.error(f"Invalid timing configuration for SprayScheduler: {e}")
            raise ValueError(f"Invalid timing configuration for SprayScheduler: {e}") from e

        if self._nozzle_config.num_nozzles == 0:
            self._logger.warning("SprayScheduler initialized with zero nozzles from NozzleConfiguration.")
        
        self._all_nozzles_cache: List[Nozzle] = self._nozzle_config.get_all_nozzles()


    def get_current_nozzle_state(self,
                                 all_managed_objects: List[ManagedObject],
                                 current_velocity_mps: Tuple[float, float]
                                 ) -> List[bool]:
        num_hw_nozzles = self._nozzle_config.num_nozzles
        if num_hw_nozzles == 0 or not self._all_nozzles_cache:
            # If no nozzles, ensure any objects marked SPRAYING are transitioned
            for obj in all_managed_objects:
                if obj.status == ObjectStatus.SPRAYING:
                    obj.status = ObjectStatus.SPRAYED
            return []

        vx_mps, vy_mps = current_velocity_mps
        nozzle_states: List[bool] = [False] * num_hw_nozzles
        # Tracks which object IDs are responsible for activating each nozzle (before safety override)
        object_initially_activating_nozzle: Dict[int, List[int]] = {i: [] for i in range(num_hw_nozzles)}

        # Step 1: Determine initial nozzle activations based on TARGETED/SPRAYING objects
        for obj in all_managed_objects:
            if obj.status not in [ObjectStatus.TARGETED, ObjectStatus.SPRAYING]:
                continue

            source_bbox_m = obj.activation_bounding_box_m
            if not source_bbox_m: continue
            
            current_act_min_x, current_act_min_y, current_act_max_x, current_act_max_y = source_bbox_m
            act_width_m = current_act_max_x - current_act_min_x
            act_length_m = current_act_max_y - current_act_min_y
            if act_width_m <= 0 or act_length_m <= 0: continue
            
            future_obj_center_x_m = obj.predicted_position_robot_m[0] + vx_mps * self._effective_latency_sec
            future_obj_center_y_m = obj.predicted_position_robot_m[1] + vy_mps * self._effective_latency_sec
            
            half_act_w = act_width_m / 2.0
            half_act_l = act_length_m / 2.0
            y_margin_m_spray = abs(vy_mps * self._spray_margin_sec)

            obj_future_min_x = future_obj_center_x_m - half_act_w
            obj_future_max_x = future_obj_center_x_m + half_act_w
            obj_future_min_y = future_obj_center_y_m - half_act_l - y_margin_m_spray
            obj_future_max_y = future_obj_center_y_m + half_act_l + y_margin_m_spray
            
            for nozzle in self._all_nozzles_cache:
                if not nozzle.bounding_box_relative_m: continue

                nz_abs_pos_x, nz_abs_pos_y = nozzle.position_m
                nz_rel_bb_min_x, nz_rel_bb_min_y, nz_rel_bb_max_x, nz_rel_bb_max_y = nozzle.bounding_box_relative_m
                
                nz_abs_bb_min_x = nz_abs_pos_x + nz_rel_bb_min_x
                nz_abs_bb_min_y = nz_abs_pos_y + nz_rel_bb_min_y
                nz_abs_bb_max_x = nz_abs_pos_x + nz_rel_bb_max_x
                nz_abs_bb_max_y = nz_abs_pos_y + nz_rel_bb_max_y

                x_overlap = (obj_future_min_x < nz_abs_bb_max_x) and \
                            (obj_future_max_x > nz_abs_bb_min_x)
                y_overlap = (obj_future_min_y < nz_abs_bb_max_y) and \
                            (obj_future_max_y > nz_abs_bb_min_y)

                if x_overlap and y_overlap:
                    nozzle_idx = nozzle.index
                    if 0 <= nozzle_idx < num_hw_nozzles:
                        nozzle_states[nozzle_idx] = True
                        if obj.track_id not in object_initially_activating_nozzle[nozzle_idx]:
                             object_initially_activating_nozzle[nozzle_idx].append(obj.track_id)
        
        # Step 2: Apply safety overrides from protected objects
        for obj in all_managed_objects:
            if obj.current_safety_margin_m > 0.0001 and obj.safety_bounding_box_m:    
                s_min_x, s_min_y, s_max_x, s_max_y = obj.safety_bounding_box_m
                s_center_x = (s_min_x + s_max_x) / 2.0
                s_center_y = (s_min_y + s_max_y) / 2.0
                s_width = s_max_x - s_min_x
                s_length = s_max_y - s_min_y

                if s_width <=0 or s_length <=0: continue

                future_s_center_x = s_center_x + vx_mps * self._effective_latency_sec
                future_s_center_y = s_center_y + vy_mps * self._effective_latency_sec

                future_keep_out_min_x = future_s_center_x - s_width / 2.0
                future_keep_out_max_x = future_s_center_x + s_width / 2.0
                future_keep_out_min_y = future_s_center_y - s_length / 2.0
                future_keep_out_max_y = future_s_center_y + s_length / 2.0

                for nozzle in self._all_nozzles_cache:
                    nozzle_idx = nozzle.index
                    if not nozzle_states[nozzle_idx]: continue # Nozzle already off, no need to check

                    if not nozzle.bounding_box_relative_m: continue
                    nz_abs_pos_x, nz_abs_pos_y = nozzle.position_m
                    nz_rel_bb_min_x, nz_rel_bb_min_y, nz_rel_bb_max_x, nz_rel_bb_max_y = nozzle.bounding_box_relative_m
                    
                    nz_abs_bb_min_x = nz_abs_pos_x + nz_rel_bb_min_x
                    nz_abs_bb_min_y = nz_abs_pos_y + nz_rel_bb_min_y
                    nz_abs_bb_max_x = nz_abs_pos_x + nz_rel_bb_max_x
                    nz_abs_bb_max_y = nz_abs_pos_y + nz_rel_bb_max_y
                    
                    x_overlap_safety = (future_keep_out_min_x < nz_abs_bb_max_x) and \
                                       (future_keep_out_max_x > nz_abs_bb_min_x)
                    y_overlap_safety = (future_keep_out_min_y < nz_abs_bb_max_y) and \
                                       (future_keep_out_max_y > nz_abs_bb_min_y)

                    if x_overlap_safety and y_overlap_safety:
                        nozzle_states[nozzle_idx] = False # Force OFF due to safety

        # Step 3: Update object statuses (SPRAYING, SPRAYED) and assigned nozzles
        for obj in all_managed_objects:
            is_obj_causing_any_active_spray_this_tick = False
            current_tick_assigned_nozzles_for_obj: List[int] = []

            for nozzle_idx in range(num_hw_nozzles):
                # Check if this nozzle is ON AND this object was one of the initial activators for it
                if nozzle_states[nozzle_idx] and \
                   obj.track_id in object_initially_activating_nozzle[nozzle_idx]:
                    is_obj_causing_any_active_spray_this_tick = True
                    current_tick_assigned_nozzles_for_obj.append(nozzle_idx)
            
            obj.assigned_nozzle_indices = sorted(list(set(current_tick_assigned_nozzles_for_obj)))

            if is_obj_causing_any_active_spray_this_tick:
                if obj.status != ObjectStatus.SPRAYING:
                    obj.status = ObjectStatus.SPRAYING
                    obj.spray_start_time = None 
                    obj.spray_end_time = None
            else: 
                if obj.status == ObjectStatus.SPRAYING:
                    obj.status = ObjectStatus.SPRAYED 
                    obj.spray_start_time = None 
                    obj.spray_end_time = None
        
        return nozzle_states