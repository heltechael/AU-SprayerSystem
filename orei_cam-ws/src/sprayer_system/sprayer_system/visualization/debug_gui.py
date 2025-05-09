import threading
import time
from typing import List, Optional, Dict, Any, Tuple, Callable
import logging
from rclpy.time import Duration

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[WARN] DebugGui: OpenCV not found, GUI functionality will be unavailable.")


from rclpy.clock import Clock
from ..common.definitions import DisplayObjectState, ObjectStatus, CameraParams, ManagedObject, Nozzle
from ..planning.nozzle_configuration import NozzleConfiguration
from ..common import geometry

STATUS_COLORS: Dict[ObjectStatus, Tuple[int, int, int]] = {
    ObjectStatus.PENDING: (0, 255, 255),      # Yellow
    ObjectStatus.TARGETED: (0, 255, 0),       # Green
    ObjectStatus.IGNORED: (0, 0, 255),        # Red
    ObjectStatus.SCHEDULED: (255, 0, 0),      # Blue
    ObjectStatus.SPRAYING: (255, 0, 255),     # Magenta
    ObjectStatus.SPRAYED: (128, 128, 128),    # Gray
    ObjectStatus.MISSED: (0, 165, 255),       # Orange
    ObjectStatus.LOST: (50, 50, 50),          # Dark Gray
    ObjectStatus.APPROACHING: (255, 255, 0),  # Cyan
}
DEFAULT_COLOR = (255, 255, 255) 

NOZZLE_MARKER_COLOR_OFF = (255, 165, 0) 
NOZZLE_MARKER_COLOR_ON = (0, 220, 0)   
NOZZLE_MARKER_RADIUS_OFF = 4
NOZZLE_MARKER_RADIUS_ON = 5 
NOZZLE_MARKER_BORDER_COLOR_ON = (0,0,0) 
NOZZLE_PATTERN_COLOR = (100, 100, 100) 
NOZZLE_PATTERN_THICKNESS = 1

ACTIVATION_LINE_COLOR_OFF = (0, 180, 180)  
ACTIVATION_LINE_COLOR_ON = NOZZLE_MARKER_COLOR_ON 
ACTIVATION_LINE_THICKNESS = 1

SCHEDULED_MARKER_COLOR = (255, 255, 255) 
SCHEDULED_MARKER_OFFSET_Y_FROM_LINE = -5 
SCHEDULED_MARKER_SCALE_FACTOR = 0.7 

SPRAY_FILL_COLOR = STATUS_COLORS[ObjectStatus.SPRAYING] 

SAFETY_ZONE_OUTLINE_COLOR = (0, 0, 180) # Darker Red for outline
SAFETY_ZONE_THICKNESS = 1
SAFETY_ZONE_FILL_COLOR = (0, 0, 100) # Dark Red Fill
SAFETY_ZONE_FILL_OPACITY = 0.5


BUTTON_HEIGHT = 30; BUTTON_SPACING = 5; BUTTON_START_Y = 5; BUTTON_WIDTH = 150
BUTTON_DEFS = [
    ((BUTTON_SPACING, BUTTON_START_Y, BUTTON_WIDTH, BUTTON_HEIGHT), "simple_weed", "Simple Weed"),
    ((BUTTON_SPACING * 2 + BUTTON_WIDTH, BUTTON_START_Y, BUTTON_WIDTH, BUTTON_HEIGHT), "spray_all", "Spray All"),
]
BUTTON_TEXT_COLOR = (0, 0, 0); BUTTON_DEFAULT_BG = (180, 180, 180); BUTTON_HIGHLIGHT_BG = (220, 220, 220)

LEADING_EDGE_COLOR = (0, 0, 255); TRAILING_EDGE_COLOR = (0, 165, 255); EDGE_LINE_THICKNESS = 1


class DebugGui:
    def __init__(self,
                 gui_config: Dict[str, Any],
                 node_config: Dict[str, Any], 
                 node_clock: Clock,
                 node_logger: logging.Logger,
                 strategy_change_callback: Callable[[str], None]):

        self._enabled = gui_config.get('enable_gui', False) and OPENCV_AVAILABLE
        self._config = gui_config
        self._node_config = node_config
        self._clock = node_clock
        self._logger = node_logger
        self._strategy_change_callback = strategy_change_callback

        self._thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._data_lock = threading.Lock()

        self._current_display_states: List[DisplayObjectState] = []
        self._current_scheduled_objects: List[ManagedObject] = []
        self._current_nozzle_config: Optional[NozzleConfiguration] = None
        self._current_target_nozzle_map: Dict[int, List[int]] = {}
        self._current_nozzle_command: List[bool] = []

        self._gps_lock = threading.Lock()
        self._latest_gps_speed: Optional[float] = None
        self._latest_gps_track: Optional[float] = None

        self._active_strategy_lock = threading.Lock()
        self._active_strategy_name: str = node_config.get('strategy', {}).get('type', 'unknown').lower()

        self._window_name = self._config.get('gui_window_name', 'Sprayer Debug')
        self._gui_width = self._config.get('gui_width', 1232)
        self._gui_height = self._config.get('gui_height', 704)
        self._update_rate = self._config.get('gui_update_rate', 30.0)
        self._wait_ms = max(1, int(1000 / self._update_rate)) if self._update_rate > 0 else 33

        self._text_color = tuple(self._config.get('gui_text_color', [255, 255, 255]))
        self._font_scale = self._config.get('gui_font_scale', 0.5)
        self._font_thickness = self._config.get('gui_font_thickness', 1)
        self._font_face = cv2.FONT_HERSHEY_SIMPLEX if OPENCV_AVAILABLE else None
        self._obj_fill_opacity = self._config.get('gui_object_fill_opacity', 0.4)
        self._spray_fill_opacity = self._config.get('gui_spray_fill_opacity', 0.6) 
        self._outline_thickness = 1
        self._outline_color = (255,255,255) 
        self._object_activation_zone_color = tuple(self._config.get('gui_object_activation_zone_color', [100, 255, 100]))


        self._cam_params: Optional[CameraParams] = None
        self._latency_sec: float = 0.0 
        self._source_width: int = self._gui_width 
        self._source_height: int = self._gui_height

        if self._enabled:
            try:
                cam_params_dict = self._node_config.get('camera_parameters', {})
                self._cam_params = CameraParams(
                    width_px=cam_params_dict['image_width_px'],
                    height_px=cam_params_dict['image_height_px'],
                    gsd_px_per_m=cam_params_dict['gsd_px_per_meter']
                )
                self._source_width = self._cam_params.width_px
                self._source_height = self._cam_params.height_px
                
                timing_config = self._node_config.get('timing', {})
                self._latency_sec = float(timing_config.get('nozzle_actuation_latency', 0.050))
                self._logger.info(f"DebugGUI Initialized. GUI Size: {self._gui_width}x{self._gui_height}. "
                                  f"Source Image: {self._source_width}x{self._source_height}. "
                                  f"Latency for viz: {self._latency_sec:.3f}s")
            except (KeyError, ValueError, TypeError) as e:
                self._logger.error(f"DebugGUI: Error parsing camera/timing params from node_config: {e}. GUI disabled.")
                self._enabled = False
        else: 
            if not OPENCV_AVAILABLE: self._logger.warn("DebugGUI disabled: OpenCV not available.")
            else: self._logger.info("DebugGUI explicitly disabled in configuration.")


    def update_active_strategy_name(self, name: str):
        with self._active_strategy_lock:
            self._active_strategy_name = name.lower()

    def start(self):
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            self._logger.warn("DebugGUI thread already started.")
            return
        
        self._logger.info("Starting DebugGUI thread...")
        self._shutdown_event.clear()
        with self._data_lock: 
            self._current_display_states.clear()
            self._current_scheduled_objects.clear()
            self._current_nozzle_config = None
            self._current_target_nozzle_map.clear()
            self._current_nozzle_command.clear()
        with self._gps_lock:
            self._latest_gps_speed = None
            self._latest_gps_track = None
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="debug_gui_thread")
        self._thread.start()

    def stop(self):
        if self._thread is not None and self._thread.is_alive():
            self._logger.info("Signaling DebugGUI thread to stop...")
            self._shutdown_event.set()
            try:
                self._thread.join(timeout=2.0) 
                if self._thread.is_alive():
                    self._logger.warn("DebugGUI thread did not exit cleanly after timeout.")
            except Exception as e:
                self._logger.error(f"Error joining DebugGUI thread: {e}")
            finally:
                self._thread = None 
                if OPENCV_AVAILABLE:
                    try:
                        cv2.destroyWindow(self._window_name)
                        cv2.waitKey(10) 
                        self._logger.info("DebugGUI window destroyed.")
                    except Exception as cv_e: 
                        self._logger.warning(f"Exception during OpenCV window destruction: {cv_e}")
        else:
            self._logger.info("DebugGUI stop called, but thread was not running or already stopped.")


    def update_from_gps(self, speed: Optional[float], track: Optional[float]):
        if not self._enabled: return
        with self._gps_lock:
            self._latest_gps_speed = speed
            self._latest_gps_track = track

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_strategy_type: Optional[str] = None
            for rect, strategy_name_code, _ in BUTTON_DEFS: 
                bx, by, bw, bh = rect
                if bx <= x < bx + bw and by <= y < by + bh:
                    clicked_strategy_type = strategy_name_code
                    break
            if clicked_strategy_type:
                self._logger.info(f"DebugGUI: Button '{clicked_strategy_type}' clicked.")
                try:
                    self._strategy_change_callback(clicked_strategy_type)
                except Exception as e: 
                    self._logger.error(f"Error in strategy change callback from GUI: {e}")

    def update_display_data(self,
                            display_states: List[DisplayObjectState],
                            scheduled_objects: List[ManagedObject],
                            nozzle_config: Optional[NozzleConfiguration],
                            target_nozzle_map: Dict[int, List[int]],
                            current_nozzle_command: List[bool]):
        if not self._enabled: return
        with self._data_lock:
            self._current_display_states = list(display_states)
            self._current_scheduled_objects = list(scheduled_objects)
            self._current_nozzle_config = nozzle_config
            self._current_target_nozzle_map = dict(target_nozzle_map)
            self._current_nozzle_command = list(current_nozzle_command)

    def _run_loop(self):
        if not OPENCV_AVAILABLE:
            self._logger.error("DebugGUI loop cannot run: OpenCV missing.")
            return

        self._logger.info(f"DebugGUI loop started. Window: '{self._window_name}'.")
        target_loop_freq = self._node_config.get('control_loop_frequency', 1.0)

        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self._window_name, self._mouse_callback)
        except Exception as e: 
             self._logger.error(f"DebugGUI: Failed to create window/set callback: {e}. Loop terminating.")
             return

        last_fps_calc_time_mono = time.monotonic()
        frame_count = 0
        display_fps = 0.0

        while not self._shutdown_event.is_set():
            current_mono_time = time.monotonic()
            with self._data_lock:
                display_states_snapshot = self._current_display_states
                scheduled_objects_snapshot = self._current_scheduled_objects 
                nozzle_config_snapshot = self._current_nozzle_config
                target_nozzle_map_snapshot = self._current_target_nozzle_map 
                nozzle_command_snapshot = self._current_nozzle_command 
            with self._gps_lock:
                current_gps_speed = self._latest_gps_speed
                current_gps_track = self._latest_gps_track
            with self._active_strategy_lock:
                active_strategy = self._active_strategy_name

            canvas = np.zeros((self._gui_height, self._gui_width, 3), dtype=np.uint8)
            obj_overlay = canvas.copy()
            spray_overlay = canvas.copy()
            safety_zone_overlay = canvas.copy()

            actually_spraying_nozzle_indices = set()
            for obj_managed in scheduled_objects_snapshot:
                if obj_managed.status == ObjectStatus.SPRAYING:
                    actually_spraying_nozzle_indices.update(obj_managed.assigned_nozzle_indices)
            
            scheduled_nozzle_indices = set()
            for obj_managed in scheduled_objects_snapshot:
                 if obj_managed.status == ObjectStatus.SCHEDULED and obj_managed.assigned_nozzle_indices:
                      scheduled_nozzle_indices.update(obj_managed.assigned_nozzle_indices)

            if nozzle_config_snapshot and self._cam_params:
                num_total_nozzles = nozzle_config_snapshot.num_nozzles
                current_command_state_per_nozzle = [False] * num_total_nozzles
                for i, cmd_state in enumerate(nozzle_command_snapshot):
                    if i < num_total_nozzles:
                        current_command_state_per_nozzle[i] = cmd_state
                
                nozzle_draw_scale_x = self._gui_width / self._cam_params.width_px if self._cam_params.width_px > 0 else 1.0
                nozzle_draw_scale_y = self._gui_height / self._cam_params.height_px if self._cam_params.height_px > 0 else 1.0


                for nozzle_obj in nozzle_config_snapshot.get_all_nozzles():
                    try:
                        nozzle_idx = nozzle_obj.index
                        is_commanded_on_this_tick = current_command_state_per_nozzle[nozzle_idx]
                        
                        gui_activation_line_y = -1
                        gui_activation_line_start_x = -1
                        gui_activation_line_end_x = -1
                        gui_activation_line_center_x = -1

                        if nozzle_obj.activation_y_m is not None and nozzle_obj.bounding_box_relative_m:
                            rel_min_x, _, rel_max_x, _ = nozzle_obj.bounding_box_relative_m
                            activation_y_ground_m = nozzle_obj.activation_y_m
                            activation_start_x_ground_m = nozzle_obj.position_m[0] + rel_min_x
                            activation_end_x_ground_m = nozzle_obj.position_m[0] + rel_max_x
                            
                            _, act_line_y_img_px = geometry.robot_ground_m_to_image_px((0.0, activation_y_ground_m), self._cam_params)
                            act_line_start_x_img_px, _ = geometry.robot_ground_m_to_image_px((activation_start_x_ground_m, 0.0), self._cam_params)
                            act_line_end_x_img_px, _   = geometry.robot_ground_m_to_image_px((activation_end_x_ground_m, 0.0), self._cam_params)

                            gui_activation_line_y = int(act_line_y_img_px * nozzle_draw_scale_y)
                            gui_activation_line_start_x = int(act_line_start_x_img_px * nozzle_draw_scale_x)
                            gui_activation_line_end_x = int(act_line_end_x_img_px * nozzle_draw_scale_x)
                            gui_activation_line_center_x = (gui_activation_line_start_x + gui_activation_line_end_x) // 2
                            
                            current_activation_line_color = ACTIVATION_LINE_COLOR_ON if is_commanded_on_this_tick else ACTIVATION_LINE_COLOR_OFF
                            cv2.line(canvas, (gui_activation_line_start_x, gui_activation_line_y), 
                                     (gui_activation_line_end_x, gui_activation_line_y), 
                                     current_activation_line_color, ACTIVATION_LINE_THICKNESS)
                        
                        marker_gui_x, marker_gui_y = -1, -1
                        if gui_activation_line_center_x != -1 and gui_activation_line_y != -1:
                            marker_gui_x, marker_gui_y = gui_activation_line_center_x, gui_activation_line_y
                        else:
                            abs_nz_physical_x_m, abs_nz_physical_y_m = nozzle_obj.position_m
                            abs_nz_img_px_x, abs_nz_img_px_y = geometry.robot_ground_m_to_image_px((abs_nz_physical_x_m, abs_nz_physical_y_m), self._cam_params)
                            marker_gui_x = int(abs_nz_img_px_x * nozzle_draw_scale_x)
                            marker_gui_y = int(abs_nz_img_px_y * nozzle_draw_scale_y)

                        if marker_gui_x != -1:
                            current_marker_radius = NOZZLE_MARKER_RADIUS_ON if is_commanded_on_this_tick else NOZZLE_MARKER_RADIUS_OFF
                            current_marker_color = NOZZLE_MARKER_COLOR_ON if is_commanded_on_this_tick else NOZZLE_MARKER_COLOR_OFF
                            cv2.circle(canvas, (marker_gui_x, marker_gui_y), current_marker_radius, current_marker_color, -1)
                            if is_commanded_on_this_tick:
                                 cv2.circle(canvas, (marker_gui_x, marker_gui_y), current_marker_radius, NOZZLE_MARKER_BORDER_COLOR_ON, 1)
                        
                        pattern_gui_pts_for_fill = None
                        if nozzle_obj.spray_pattern_relative_m:
                            abs_nz_physical_x_m, abs_nz_physical_y_m = nozzle_obj.position_m
                            pattern_vertices_gui_px = []
                            for rel_m_x, rel_m_y in nozzle_obj.spray_pattern_relative_m:
                                abs_pattern_vertex_x_m = abs_nz_physical_x_m + rel_m_x
                                abs_pattern_vertex_y_m = abs_nz_physical_y_m + rel_m_y
                                abs_pattern_vertex_img_px_x, abs_pattern_vertex_img_px_y = geometry.robot_ground_m_to_image_px(
                                    (abs_pattern_vertex_x_m, abs_pattern_vertex_y_m), self._cam_params
                                )
                                pattern_vertices_gui_px.append(
                                    (int(abs_pattern_vertex_img_px_x * nozzle_draw_scale_x), 
                                     int(abs_pattern_vertex_img_px_y * nozzle_draw_scale_y))
                                )
                            if pattern_vertices_gui_px:
                                pattern_gui_pts_for_fill = np.array(pattern_vertices_gui_px, dtype=np.int32).reshape((-1,1,2))
                                cv2.polylines(canvas, [pattern_gui_pts_for_fill], isClosed=True, color=NOZZLE_PATTERN_COLOR, thickness=NOZZLE_PATTERN_THICKNESS)
                        
                        if nozzle_idx in scheduled_nozzle_indices and gui_activation_line_center_x != -1 and gui_activation_line_y != -1:
                            s_text_size, _ = cv2.getTextSize("S", self._font_face, self._font_scale * SCHEDULED_MARKER_SCALE_FACTOR, self._font_thickness)
                            s_text_w, s_text_h = s_text_size
                            s_marker_pos_x = gui_activation_line_center_x - s_text_w // 2
                            s_marker_pos_y = gui_activation_line_y + SCHEDULED_MARKER_OFFSET_Y_FROM_LINE + s_text_h // 2
                            cv2.putText(canvas, "S", (s_marker_pos_x, s_marker_pos_y), self._font_face,
                                        self._font_scale * SCHEDULED_MARKER_SCALE_FACTOR, 
                                        SCHEDULED_MARKER_COLOR, self._font_thickness, cv2.LINE_AA)
                        
                        if pattern_gui_pts_for_fill is not None and nozzle_idx in actually_spraying_nozzle_indices:
                            cv2.fillPoly(spray_overlay, [pattern_gui_pts_for_fill], SPRAY_FILL_COLOR)

                    except Exception as e: 
                        self._logger.warning(f"DebugGUI: Error drawing nozzle {nozzle_obj.index}: {e}", throttle_duration_sec=10)
            
            for state in display_states_snapshot:
                if state.safety_bbox_gui_px:
                    try:
                        s_x1, s_y1, s_x2, s_y2 = map(int, state.safety_bbox_gui_px)
                        if s_x2 > s_x1 and s_y1 < s_y2 :
                             cv2.rectangle(safety_zone_overlay, (s_x1, s_y1), (s_x2, s_y2), 
                                           SAFETY_ZONE_FILL_COLOR, -1) # Fill on overlay
                             cv2.rectangle(canvas, (s_x1, s_y1), (s_x2, s_y2), 
                                           SAFETY_ZONE_OUTLINE_COLOR, SAFETY_ZONE_THICKNESS) # Outline on canvas
                    except Exception as e_sz:
                         self._logger.warning(f"DebugGUI: Error drawing safety zone for obj {state.track_id}: {e_sz}", throttle_duration_sec=10)


            for state in display_states_snapshot:
                try:
                    gui_pred_center_x = int(state.predicted_center_image_px[0])
                    gui_pred_center_y = int(state.predicted_center_image_px[1])

                    orig_w_px_scaled = int(state.original_bbox_dims_px[0])
                    orig_h_px_scaled = int(state.original_bbox_dims_px[1])

                    gui_x1_orig = max(0, gui_pred_center_x - orig_w_px_scaled // 2)
                    gui_y1_orig = max(0, gui_pred_center_y - orig_h_px_scaled // 2)
                    gui_x2_orig = min(self._gui_width -1, gui_pred_center_x + orig_w_px_scaled // 2)
                    gui_y2_orig = min(self._gui_height-1, gui_pred_center_y + orig_h_px_scaled // 2)

                    if gui_x2_orig > gui_x1_orig and gui_y2_orig > gui_y1_orig :
                        fill_color = STATUS_COLORS.get(state.status, DEFAULT_COLOR)
                        cv2.rectangle(obj_overlay, (gui_x1_orig, gui_y1_orig), (gui_x2_orig, gui_y2_orig), fill_color, -1) 
                        cv2.rectangle(canvas, (gui_x1_orig, gui_y1_orig), (gui_x2_orig, gui_y2_orig), self._outline_color, self._outline_thickness) 
                        cv2.line(canvas, (gui_x1_orig, gui_y1_orig), (gui_x2_orig, gui_y1_orig), LEADING_EDGE_COLOR, EDGE_LINE_THICKNESS)
                        cv2.line(canvas, (gui_x1_orig, gui_y2_orig), (gui_x2_orig, gui_y2_orig), TRAILING_EDGE_COLOR, EDGE_LINE_THICKNESS)

                        act_w_px_scaled = int(state.activation_bbox_dims_px[0])
                        
                        if abs(state.activation_bbox_dims_px[0] - state.original_bbox_dims_px[0]) > 1e-2: 
                            gui_x1_act = max(0, gui_pred_center_x - act_w_px_scaled // 2)
                            gui_y1_act = gui_y1_orig 
                            gui_x2_act = min(self._gui_width - 1, gui_pred_center_x + act_w_px_scaled // 2)
                            gui_y2_act = gui_y2_orig 

                            if gui_x2_act > gui_x1_act and gui_y2_act > gui_y1_act:
                                cv2.rectangle(canvas, (gui_x1_act, gui_y1_act), (gui_x2_act, gui_y2_act),
                                              self._object_activation_zone_color, self._outline_thickness)
                        
                        text_y_pos = gui_y1_orig - 5 if gui_y1_orig > 15 else gui_y2_orig + 15
                        cv2.putText(canvas, state.label, (gui_x1_orig, text_y_pos), self._font_face,
                                    self._font_scale, self._text_color, self._font_thickness, cv2.LINE_AA)
                except Exception as e: 
                    self._logger.warning(f"DebugGUI: Error drawing object {state.track_id}: {e}", throttle_duration_sec=10)
            
            # Blend overlays
            cv2.addWeighted(safety_zone_overlay, SAFETY_ZONE_FILL_OPACITY, canvas, 1.0, 0, canvas)
            cv2.addWeighted(obj_overlay, self._obj_fill_opacity, canvas, 1.0, 0, canvas)
            cv2.addWeighted(spray_overlay, self._spray_fill_opacity, canvas, 1.0, 0, canvas)

            text_y_pos = BUTTON_START_Y + BUTTON_HEIGHT + BUTTON_SPACING * 2
            line_height = 18 

            for rect_def, strategy_code, display_name in BUTTON_DEFS:
                bx, by, bw, bh = rect_def
                current_button_bg_color = BUTTON_HIGHLIGHT_BG if strategy_code == active_strategy else BUTTON_DEFAULT_BG
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), current_button_bg_color, -1)
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), BUTTON_TEXT_COLOR, 1)
                text_size, _ = cv2.getTextSize(display_name, self._font_face, self._font_scale, self._font_thickness)
                text_w, text_h = text_size
                cv2.putText(canvas, display_name, (bx + (bw-text_w)//2, by + (bh+text_h)//2), 
                            self._font_face, self._font_scale, BUTTON_TEXT_COLOR, self._font_thickness, cv2.LINE_AA)
            
            info_texts = [
                f"GUI FPS: {display_fps:.1f} | Node Loop Target: {target_loop_freq:.1f}Hz",
                f"Managed Objects: {len(display_states_snapshot)}",
                f"GPS Speed: {current_gps_speed:.2f}m/s" if current_gps_speed is not None else "GPS Speed: N/A",
                f"GPS Track: {current_gps_track:.1f}deg" if current_gps_track is not None else "GPS Track: N/A",
                f"Strategy: {active_strategy.replace('_',' ').title()}"
            ]
            for text_line in info_texts:
                cv2.putText(canvas, text_line, (10, text_y_pos), self._font_face, self._font_scale, self._text_color, self._font_thickness)
                text_y_pos += line_height

            if not display_states_snapshot:
                no_obj_display_text = "No objects managed/visible"
                text_size_no_obj, _ = cv2.getTextSize(no_obj_display_text, self._font_face, self._font_scale*1.2, self._font_thickness)
                text_w_no_obj, text_h_no_obj = text_size_no_obj
                cv2.putText(canvas, no_obj_display_text, 
                            ((self._gui_width - text_w_no_obj) // 2, (self._gui_height + text_h_no_obj) // 2), 
                            self._font_face, self._font_scale*1.2, (0,165,255), self._font_thickness)

            try:
                cv2.imshow(self._window_name, canvas)
                key = cv2.waitKey(self._wait_ms) & 0xFF
                if key == 27 or key == ord('q'):
                    self._logger.info("DebugGUI: Close requested via keypress.")
                    self._shutdown_event.set()
                if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._logger.info("DebugGUI: Window closed by user (e.g., 'X' button).")
                    self._shutdown_event.set()
            except cv2.error as e:
                if "NULL window" in str(e) or "Invalid window" in str(e) or "checkWindowIsValid" in str(e).lower():
                     self._logger.warning(f"DebugGUI: OpenCV window became invalid or was closed externally. Loop terminating. Error: {e}")
                else: self._logger.error(f"DebugGUI: OpenCV error during imshow/waitKey: {e}. Loop terminating.")
                self._shutdown_event.set()
            except Exception as e: 
                self._logger.error(f"DebugGUI: Unexpected error in display/event handling: {e}. Loop terminating.", exc_info=True)
                self._shutdown_event.set()

            frame_count += 1
            elapsed_time_for_fps_calc = current_mono_time - last_fps_calc_time_mono
            if elapsed_time_for_fps_calc >= 1.0:
                display_fps = frame_count / elapsed_time_for_fps_calc
                frame_count = 0
                last_fps_calc_time_mono = current_mono_time
        
        self._logger.info("DebugGUI loop finished.")