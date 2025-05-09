import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

from rclpy.time import Time, Duration
from rclpy.clock import Clock
from std_msgs.msg import Header
from vision_interfaces.msg import TrackedObject as TrackedObjectMsg

from ..common.definitions import ManagedObject, ObjectStatus, DisplayObjectState, CameraParams
from ..motion import MotionModel
from ..common import geometry 
from ..strategy import BaseSprayStrategy

class ObjectManager:
    def __init__(self,
                 config: Dict[str, Any],
                 motion_model: MotionModel,
                 cam_params: CameraParams,
                 initial_strategy: BaseSprayStrategy, 
                 logger: logging.Logger,
                 clock: Clock):
        self._config = config
        self._motion_model = motion_model
        self._cam_params = cam_params
        self._strategy = initial_strategy 
        self._logger = logger
        self._clock = clock
        self._objects: Dict[int, ManagedObject] = {}
        self._lock = threading.RLock()
        self._last_prediction_mono_time: Optional[float] = None

        om_config = self._config.get('object_management', {})
        self._activation_zone_object_width_px = om_config.get('activation_zone_object_width_px', -1.0)
        if self._activation_zone_object_width_px > 0:
            logger.info(f"ObjectManager: Object activation zone width will be shrunk to {self._activation_zone_object_width_px:.1f} px if smaller than original.")
        else:
            logger.info("ObjectManager: Object activation zone width shrinkage is disabled.")

        self._logger.info(f"ObjectManager initialized. Initial Strategy: {self._strategy.__class__.__name__}")
        self._logger.info(f"  Transforms based on image size: {self._cam_params.width_px}x{self._cam_params.height_px}")
        self._logger.info(f"  Using simplified GSD model: {1.0/self._cam_params.m_per_px:.1f} px/m")

    def set_strategy(self, new_strategy: BaseSprayStrategy):
        if new_strategy is None:
            self._logger.error("Attempted to set a null strategy.")
            return

        with self._lock: 
            old_strategy_name = self._strategy.__class__.__name__
            new_strategy_name = new_strategy.__class__.__name__
            if old_strategy_name == new_strategy_name:
                self._logger.info(f"Strategy is already set to {new_strategy_name}. No change.")
                return

            self._logger.info(f"Switching strategy from {old_strategy_name} to {new_strategy_name}...")
            self._strategy = new_strategy

            re_eval_statuses = [ObjectStatus.PENDING, ObjectStatus.TARGETED, ObjectStatus.IGNORED]
            num_re_evaluated = 0
            status_changes = 0
            keys_to_iterate = list(self._objects.keys())

            for track_id in keys_to_iterate:
                if track_id not in self._objects: continue 
                obj = self._objects[track_id]
                if obj.status in re_eval_statuses:
                    try:
                        old_status = obj.status
                        decision_result = self._strategy.decide(obj) 
                        new_status = ObjectStatus.TARGETED if decision_result else ObjectStatus.IGNORED
                        obj.status = new_status
                        num_re_evaluated += 1
                        if old_status != new_status:
                            status_changes += 1
                        
                        safety_m = self._strategy.get_safety_zone_m(obj)
                        obj.update_bounding_boxes(self._cam_params, self._activation_zone_object_width_px, safety_m)

                    except Exception as strategy_e:
                        self._logger.error(f"  Error re-applying new strategy to target {track_id}: {strategy_e}")
                        obj.status = ObjectStatus.PENDING 
                        obj.update_bounding_boxes(self._cam_params, self._activation_zone_object_width_px, 0.0)


            self._logger.info(f"Strategy switched to {new_strategy_name}. Re-evaluated {num_re_evaluated} objects ({status_changes} status changes).")

    def update_from_tracking(self, tracked_objects: List[TrackedObjectMsg], message_header: Header):
        message_time = Time.from_msg(message_header.stamp)

        with self._lock: 
            updated_ids = set()
            strategy_decisions_made = 0
            status_changes = 0

            for obj_msg in tracked_objects:
                track_id = obj_msg.track_id
                updated_ids.add(track_id)

                center_img_px = (obj_msg.center_x, obj_msg.center_y)
                bbox_img_px = (obj_msg.bbox_x, obj_msg.bbox_y, obj_msg.bbox_width, obj_msg.bbox_height)

                try:
                    position_ground_m = geometry.image_px_to_robot_ground_m(center_img_px, self._cam_params)
                    size_ground_m = geometry.estimate_object_size_m(bbox_img_px, self._cam_params)
                except Exception as e:
                    self._logger.error(f"Error transforming track {track_id} coordinates: {e}. Skipping.")
                    continue

                current_vision_class_name = obj_msg.class_name
                current_vision_confidence = obj_msg.confidence

                existing_object = self._objects.get(track_id)
                safety_m = 0.0

                if existing_object:
                    obj = existing_object
                    obj.last_seen_time = message_time
                    obj.bbox_image = bbox_img_px 
                    obj.position_robot_m = position_ground_m
                    obj.size_m = size_ground_m
                    obj.predicted_position_robot_m = position_ground_m 
                    
                    strategy_input_changed = (
                        obj.last_vision_class_name != current_vision_class_name or
                        abs(obj.last_vision_confidence - current_vision_confidence) > 0.05 
                    )

                    obj.last_vision_class_name = current_vision_class_name
                    obj.last_vision_confidence = current_vision_confidence
                    obj.confidence = current_vision_confidence 
                    obj.class_name = current_vision_class_name
                    
                    if obj.status == ObjectStatus.PENDING or \
                       (strategy_input_changed and obj.status in [ObjectStatus.TARGETED, ObjectStatus.IGNORED]):
                        try:
                            old_status = obj.status
                            decision_result = self._strategy.decide(obj)
                            new_status = ObjectStatus.TARGETED if decision_result else ObjectStatus.IGNORED
                            obj.status = new_status
                            strategy_decisions_made += 1
                            if old_status != new_status: status_changes += 1
                        except Exception as strategy_e:
                            self._logger.error(f"Error applying strategy to target {track_id}: {strategy_e}")
                            obj.status = ObjectStatus.PENDING
                    
                    safety_m = self._strategy.get_safety_zone_m(obj)
                    obj.update_bounding_boxes(self._cam_params, self._activation_zone_object_width_px, safety_m)

                    if obj.status == ObjectStatus.LOST: 
                         obj.status = ObjectStatus.PENDING
                         self._logger.info(f"Reacquired object {track_id}, set to PENDING.")
                else: 
                    new_object = ManagedObject(
                        track_id=track_id, class_id=obj_msg.class_id,
                        class_name=current_vision_class_name, last_seen_time=message_time,
                        bbox_image=bbox_img_px, confidence=current_vision_confidence,
                        last_vision_class_name=current_vision_class_name,
                        last_vision_confidence=current_vision_confidence,
                        position_robot_m=position_ground_m,
                        predicted_position_robot_m=position_ground_m, 
                        size_m=size_ground_m,
                        status=ObjectStatus.PENDING, created_time=self._clock.now()
                    )
                    
                    try: 
                        decision_result = self._strategy.decide(new_object)
                        initial_status = ObjectStatus.TARGETED if decision_result else ObjectStatus.IGNORED
                        new_object.status = initial_status
                        strategy_decisions_made += 1
                    except Exception as strategy_e:
                        self._logger.error(f"Error applying initial strategy to {track_id}: {strategy_e}")
                        new_object.status = ObjectStatus.PENDING
                    
                    safety_m = self._strategy.get_safety_zone_m(new_object)
                    new_object.update_bounding_boxes(self._cam_params, self._activation_zone_object_width_px, safety_m)
                    self._objects[track_id] = new_object


    def _is_offscreen_m(self, predicted_pos_m: Tuple[float, float], size_m: Tuple[float, float]) -> bool:
        max_lateral_m = 2.5; max_forward_m = 1.5; max_behind_m = 4.0
        pred_x, pred_y = predicted_pos_m; half_width = size_m[0] / 2.0; half_length = size_m[1] / 2.0
        if (pred_x + half_width) < -max_lateral_m or (pred_x - half_width) > max_lateral_m: return True
        if (pred_y + half_length) < -max_behind_m or (pred_y - half_length) > max_forward_m: return True
        return False

    def update_predictions_and_get_display_state(self) -> List[DisplayObjectState]:
        current_mono_time = time.monotonic()
        display_list = []; ids_to_remove = []
        dt = 0.0
        if self._last_prediction_mono_time is not None: 
            dt = max(0.0, min(current_mono_time - self._last_prediction_mono_time, 0.1))
        self._last_prediction_mono_time = current_mono_time
        
        vx_mps, vy_mps = self._motion_model.get_velocity_mps()
        dx_m = vx_mps * dt
        dy_m = vy_mps * dt
        
        gui_width = self._config.get('debug',{}).get('gui_width', self._cam_params.width_px)
        gui_height = self._config.get('debug',{}).get('gui_height', self._cam_params.height_px)
        scale_x = gui_width / self._cam_params.width_px if self._cam_params.width_px > 0 else 1.0
        scale_y = gui_height / self._cam_params.height_px if self._cam_params.height_px > 0 else 1.0

        with self._lock:
            keys_to_iterate = list(self._objects.keys())
            for track_id in keys_to_iterate:
                 if track_id not in self._objects: continue
                 obj = self._objects[track_id]
                 
                 pred_x_m = obj.predicted_position_robot_m[0] + dx_m
                 pred_y_m = obj.predicted_position_robot_m[1] + dy_m
                 obj.predicted_position_robot_m = (pred_x_m, pred_y_m)

                 safety_m = 0.0
                 if self._strategy:
                     try:
                         safety_m = self._strategy.get_safety_zone_m(obj)
                     except Exception as e_strat:
                         self._logger.error(f"Error getting safety zone for obj {obj.track_id} from strategy: {e_strat}")
                 
                 obj.update_bounding_boxes(self._cam_params, 
                                           self._activation_zone_object_width_px,
                                           safety_margin_m=safety_m)

                 if self._is_offscreen_m(obj.predicted_position_robot_m, obj.size_m): 
                     ids_to_remove.append(track_id)
                     continue
                 
                 pred_center_px_unscaled: Tuple[float, float]
                 try: 
                     pred_center_px_unscaled = geometry.robot_ground_m_to_image_px(obj.predicted_position_robot_m, self._cam_params)
                 except Exception as e: 
                     self._logger.error(f"Error transforming {track_id} to pixels for display: {e}")
                     continue
                
                 # Scale for GUI
                 pred_center_gui_px = (pred_center_px_unscaled[0] * scale_x, pred_center_px_unscaled[1] * scale_y)

                 original_dims_gui_px = (obj.bbox_image[2] * scale_x, obj.bbox_image[3] * scale_y) 

                 activation_width_px_unscaled = obj.bbox_image[2]
                 if self._activation_zone_object_width_px is not None and \
                    0 < self._activation_zone_object_width_px < activation_width_px_unscaled:
                     activation_width_px_unscaled = self._activation_zone_object_width_px
                 
                 activation_dims_gui_px = (activation_width_px_unscaled * scale_x, obj.bbox_image[3] * scale_y)

                 safety_bbox_gui_px_val: Optional[Tuple[float,float,float,float]] = None
                 if obj.safety_bounding_box_m:
                     s_min_m_x, s_min_m_y, s_max_m_x, s_max_m_y = obj.safety_bounding_box_m
                     s_min_px_unscaled = geometry.robot_ground_m_to_image_px((s_min_m_x, s_min_m_y), self._cam_params)
                     s_max_px_unscaled = geometry.robot_ground_m_to_image_px((s_max_m_x, s_max_m_y), self._cam_params)
                     safety_bbox_gui_px_val = (
                         s_min_px_unscaled[0] * scale_x, s_min_px_unscaled[1] * scale_y,
                         s_max_px_unscaled[0] * scale_x, s_max_px_unscaled[1] * scale_y
                     )

                 label = (f"{obj.track_id}:{obj.class_name}:{obj.confidence:.2f}")
                 display_list.append(DisplayObjectState(
                     track_id=obj.track_id, 
                     predicted_center_image_px=pred_center_gui_px, # Already scaled
                     original_bbox_dims_px=original_dims_gui_px,   # Already scaled
                     activation_bbox_dims_px=activation_dims_gui_px, # Already scaled
                     label=label, 
                     status=obj.status,
                     safety_bbox_gui_px=safety_bbox_gui_px_val # Already scaled
                 ))

            if ids_to_remove:
                num_removed = 0
                for track_id in ids_to_remove:
                    if track_id in self._objects: del self._objects[track_id]; num_removed += 1
                if num_removed > 0: self._logger.debug(f"Removed {num_removed} off-screen objects.")
        return display_list

    def get_all_managed_objects(self) -> List[ManagedObject]:
        with self._lock:
            for obj in self._objects.values():
                 safety_m = 0.0
                 if self._strategy:
                     try:
                         safety_m = self._strategy.get_safety_zone_m(obj)
                     except Exception as e_strat:
                         self._logger.error(f"Error getting safety zone for obj {obj.track_id} from strategy: {e_strat}")
                 
                 if obj.bounding_box_m is None or obj.activation_bounding_box_m is None or \
                    (safety_m > 0 and obj.safety_bounding_box_m is None) or \
                     abs(obj.current_safety_margin_m - safety_m) > 1e-4 :
                     obj.update_bounding_boxes(self._cam_params, 
                                               self._activation_zone_object_width_px,
                                               safety_margin_m=safety_m)
            return list(self._objects.values())

    def get_targets_by_status(self, statuses: List[ObjectStatus]) -> List[ManagedObject]:
        matching_objects = []
        all_objects = self.get_all_managed_objects() 
        with self._lock: 
            for obj in all_objects:
                if obj.status in statuses:
                    matching_objects.append(obj)
        return matching_objects