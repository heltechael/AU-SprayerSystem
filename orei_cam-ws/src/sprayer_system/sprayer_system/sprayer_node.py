# ENTRY POINT AND MAIN FILE

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.exceptions import ParameterException

from vision_interfaces.msg import TrackingResults
from gps_msgs.msg import GPSFix
from std_msgs.msg import String as StringMsg 

from .common.definitions import DisplayObjectState, CameraParams, ObjectStatus, ManagedObject
from .visualization import DebugGui, OPENCV_AVAILABLE
from .motion import MotionModel
from .object_management import ObjectManager
from .strategy import create_spraying_strategy, BaseSprayStrategy
from .planning import NozzleConfiguration, NozzleMapper, SprayScheduler
from .hardware_interface import create_hardware_interface, BaseHardwareInterface
from .hardware_interface.serial_relay_driver import SerialRelayInterface

import traceback
import threading
from typing import Dict, Any, List, Optional, Tuple
import time
import sys

class SprayerNode(Node):
    def __init__(self):
        super().__init__('sprayer_control_node')
        self.get_logger().info(f"Initializing {self.get_name()}...")
        self.config: Dict[str, Any] = {}
        self._log_messages = False
        self.debug_gui: Optional[DebugGui] = None
        self.motion_model: Optional[MotionModel] = None
        self.strategy: Optional[BaseSprayStrategy] = None
        self.object_manager: Optional[ObjectManager] = None
        self.nozzle_config: Optional[NozzleConfiguration] = None
        self.nozzle_mapper: Optional[NozzleMapper] = None
        self.control_timer: Optional[rclpy.timer.Timer] = None
        self.tracking_subscription = None
        self.gps_subscription = None
        self.spraying_state_subscription = None
        self.camera_params: Optional[CameraParams] = None
        self._strategy_update_lock = threading.Lock()
        self.spray_scheduler: Optional[SprayScheduler] = None
        self.hardware_interface: Optional[BaseHardwareInterface] = None
        self._default_strategy_type: Optional[str] = None


        try:
            self._declare_parameters()
            self.config = self._load_parameters()
            self._log_messages = self.config.get('log_received_messages', False)
            self._default_strategy_type = self.config.get('strategy', {}).get('type')
            if not self._default_strategy_type:
                self.get_logger().warn("Default strategy type not found in config. Fallback may not work as expected.")

            self._validate_and_create_cam_params(self.config.get('camera_parameters', {}))

            gui_enabled = self.config.get('debug', {}).get('enable_gui', False)
            if gui_enabled:
                if not OPENCV_AVAILABLE:
                    self.get_logger().warn("Debug GUI enabled, but OpenCV missing. GUI disabled.")
                else:
                    debug_config = self.config.get('debug', {})
                    self.debug_gui = DebugGui(
                        gui_config=debug_config, node_config=self.config,
                        node_clock=self.get_clock(), node_logger=self.get_logger(),
                        strategy_change_callback=self.request_strategy_change
                    )
            self.get_logger().info('Parameters loaded and validated successfully.')
        except (ParameterException, ValueError) as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Parameter/Validation Error: {e}\n{error_trace}")
            rclpy.try_shutdown()
            sys.exit(1)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Unexpected Parameter Setup Error: {e}\n{error_trace}")
            rclpy.try_shutdown()
            sys.exit(1)

        try:
            if not self.camera_params:
                 raise RuntimeError("CameraParams initialization failed, cannot proceed.")

            self.motion_model = MotionModel(
                config=self.config.get('motion_model', {}), cam_params=self.camera_params,
                logger=self.get_logger(), clock=self.get_clock()
            )
            self.strategy = create_spraying_strategy(
                strategy_config=self.config.get('strategy', {}),
                logger=self.get_logger(), clock=self.get_clock()
            )
            if self.strategy is None:
                raise RuntimeError("Failed to create initial spraying strategy.")

            layout_config = self.config.get('nozzle_layout', {})
            self.nozzle_config = NozzleConfiguration(layout_config, self.camera_params, self.get_logger())

            if self.nozzle_config.num_nozzles == 0:
                self.get_logger().error("NO NOZZLES CONFIGURED. Spraying functionality will be severely limited or disabled.")
            elif self.nozzle_config.num_nozzles > SerialRelayInterface.MAX_SUPPORTED_NOZZLES:
                self.get_logger().error(
                    f"Configured nozzle count ({self.nozzle_config.num_nozzles}) exceeds hardware "
                    f"limit ({SerialRelayInterface.MAX_SUPPORTED_NOZZLES}). "
                    f"Only the first {SerialRelayInterface.MAX_SUPPORTED_NOZZLES} nozzles can be controlled."
                )

            self.nozzle_mapper = NozzleMapper(self.nozzle_config, self.get_logger())
            self.object_manager = ObjectManager(
                config=self.config,
                motion_model=self.motion_model,
                cam_params=self.camera_params,
                initial_strategy=self.strategy,
                logger=self.get_logger(),
                clock=self.get_clock()
            )
            timing_config = self.config.get('timing', {})
            self.spray_scheduler = SprayScheduler(
                timing_config=timing_config, nozzle_config=self.nozzle_config,
                logger=self.get_logger(), clock=self.get_clock()
            )

            hw_config = self.config.get('hardware_interface', {})
            hw_type = hw_config.get('type', 'dummy')
            self.hardware_interface = create_hardware_interface(hw_type, hw_config, self.get_logger())
            if self.hardware_interface:
                if not self.hardware_interface.connect():
                    self.get_logger().error(f"Failed to connect to hardware '{hw_type}'. No hardware control.")
            else:
                self.get_logger().warning(f"HW interface '{hw_type}' not created or dummy. No hardware control.")
            self.get_logger().info("Core components initialized.")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Component Initialization Error: {e}\n{error_trace}")
            rclpy.try_shutdown()
            sys.exit(1)

        self._setup_communication()
        self._setup_control_loop()
        if self.debug_gui:
            self.debug_gui.start()
        self.get_logger().info(f"{self.get_name()} initialization complete.")

    def _declare_parameters(self):
        self.declare_parameter('input_topic', '/vision_system/tracked_objects')
        self.declare_parameter('gps_topic', '/gpsfix')
        self.declare_parameter('spraying_state_topic', '/sprayingState')
        self.declare_parameter('control_loop_frequency', 128.0)
        self.declare_parameter('log_received_messages', False)

        # Camera
        self.declare_parameter('camera_parameters.image_width_px', 2464)
        self.declare_parameter('camera_parameters.image_height_px', 2144)
        self.declare_parameter('camera_parameters.gsd_px_per_meter', 2209.86)

        # Motion
        self.declare_parameter('motion_model.playback_speed_factor', 1.0)
        self.declare_parameter('motion_model.gps_staleness_threshold_sec', 1.0)

        # Debugging
        self.declare_parameter('debug.enable_gui', True)
        self.declare_parameter('debug.gui_update_rate', 128.0)
        self.declare_parameter('debug.gui_window_name', "Sprayer Debug")
        self.declare_parameter('debug.gui_width', 1232)
        self.declare_parameter('debug.gui_height', 1067)
        self.declare_parameter('debug.gui_text_color', [255, 255, 255])
        self.declare_parameter('debug.gui_font_scale', 0.5)
        self.declare_parameter('debug.gui_font_thickness', 1)
        self.declare_parameter('debug.gui_object_fill_opacity', 0.4)
        self.declare_parameter('debug.gui_spray_fill_opacity', 0.6)
        self.declare_parameter('debug.gui_object_activation_zone_color', [100, 255, 100])

        # Strategy
        self.declare_parameter('strategy.type', 'simple_weed')
        self.declare_parameter('strategy.simple_weed.crop_class_names', ['BEAVA', 'ZEAMX', 'SOLTU'])
        self.declare_parameter('strategy.simple_weed.min_confidence', 0.0)
        self.declare_parameter('strategy.simple_weed.min_target_coverage_ratio', 1.0)
        self.declare_parameter('strategy.simple_weed.max_nontarget_overspray_ratio', 1.0)
        self.declare_parameter('strategy.simple_weed.safety_zone_in_cm', 0.0)

        self.declare_parameter('strategy.spray_all.min_confidence', 0.0)
        self.declare_parameter('strategy.spray_all.min_target_coverage_ratio', 1.0)
        self.declare_parameter('strategy.spray_all.max_nontarget_overspray_ratio', 1.0)

        self.declare_parameter('strategy.macro_strategy.confidence_threshold', 0.4)
        self.declare_parameter('strategy.macro_strategy.size_threshold_m', 0.04)
        self.declare_parameter('strategy.macro_strategy.min_target_coverage_ratio', 0.9)
        self.declare_parameter('strategy.macro_strategy.max_nontarget_overspray_ratio', 0.1)

        # Nozzle Layout
        self.declare_parameter('nozzle_layout.num_nozzles', 25)
        self.declare_parameter('nozzle_layout.calibration_file', "")
        self.declare_parameter('nozzle_layout.placeholder', True)
        self.declare_parameter('nozzle_layout.default_spacing_cm', 4.0)
        self.declare_parameter('nozzle_layout.default_spray_width_cm', 4.0)
        self.declare_parameter('nozzle_layout.default_spray_length_cm', 1.0)
        self.declare_parameter('nozzle_layout.default_boom_y_position_m', 0.35)
        self.declare_parameter('nozzle_layout.activation_zone_target_width_px', -1.0)
        self.declare_parameter('nozzle_layout.activation_zone_target_height_px', -1.0)
        # Timing
        self.declare_parameter('timing.nozzle_actuation_latency', 0.1)
        self.declare_parameter('timing.spray_margin_time', 0.00)

        # Object Management
        self.declare_parameter('object_management.keep_lost_objects_on_screen', False)
        self.declare_parameter('object_management.activation_zone_object_width_px', -1.0)

        # Hardware Interface
        self.declare_parameter('hardware_interface.type', 'serial_relay')
        self.declare_parameter('hardware_interface.serial_relay.port', '/dev/ttyNC0')
        self.declare_parameter('hardware_interface.serial_relay.baudrate', 9600)

    def _load_parameters(self) -> dict:
        param_names = self._parameters.keys()
        config = {}
        for param_full_name in param_names:
            param_obj = self.get_parameter(param_full_name)
            if param_obj.value is not None:
                keys = param_full_name.split('.')
                d = config
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = param_obj.value

        if not config.get('input_topic'): raise ParameterException("'input_topic' is missing or empty.")
        if config.get('control_loop_frequency', 0.0) <= 0:
            raise ParameterException("'control_loop_frequency' must be positive.")
        return config

    def _validate_and_create_cam_params(self, cam_config: Dict[str, Any]):
        width = cam_config.get('image_width_px')
        height = cam_config.get('image_height_px')
        gsd = cam_config.get('gsd_px_per_meter')
        if not (isinstance(width, int) and width > 0):
            raise ValueError(f"Invalid 'camera_parameters.image_width_px': {width}")
        if not (isinstance(height, int) and height > 0):
            raise ValueError(f"Invalid 'camera_parameters.image_height_px': {height}")
        if not (isinstance(gsd, (int, float)) and gsd > 0):
            raise ValueError(f"Invalid 'camera_parameters.gsd_px_per_meter': {gsd}")
        self.camera_params = CameraParams(width_px=width, height_px=height, gsd_px_per_m=float(gsd))
        self.get_logger().info(
            f"CameraParams created: {width}x{height}, GSD={gsd:.2f} px/m "
            f"({self.camera_params.m_per_px * 1000:.2f} mm/px)"
        )

    def _setup_communication(self):
        input_topic = self.config['input_topic']
        gps_topic = self.config.get('gps_topic', '/gpsfix')
        spraying_state_topic = self.config.get('spraying_state_topic', '/sprayingState')


        vision_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
            depth=5, durability=DurabilityPolicy.VOLATILE
        )
        gps_qos = qos_profile_sensor_data

        spraying_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL 
        )


        self.tracking_subscription = self.create_subscription(
            TrackingResults, input_topic, self.tracking_results_callback, vision_qos
        )
        self.get_logger().info(f"Subscribed to Vision: '{input_topic}'")

        self.gps_subscription = self.create_subscription(
            GPSFix, gps_topic, self._gps_fix_callback, gps_qos
        )
        self.get_logger().info(f"Subscribed to GPS: '{gps_topic}'")

        self.spraying_state_subscription = self.create_subscription(
            StringMsg, spraying_state_topic, self._spraying_state_callback, spraying_state_qos
        )
        self.get_logger().info(f"Subscribed to Spraying State Control: '{spraying_state_topic}'")


    def _gps_fix_callback(self, msg: GPSFix):
        if self.motion_model:
            self.motion_model.update_from_gps(msg)
        if self.debug_gui:
            speed = msg.speed if msg.speed == msg.speed else None
            track = msg.track if msg.track == msg.track else None
            self.debug_gui.update_from_gps(speed, track)

    def tracking_results_callback(self, msg: TrackingResults):
        msg_time = Time.from_msg(msg.header.stamp)
        if msg_time.nanoseconds == 0:
            self.get_logger().warn("TrackingResults received with zero timestamp.")

        if self._log_messages:
            self.get_logger().info(
                f"TrackingResults: TS={msg_time.nanoseconds/1e9:.3f}, Objs={len(msg.tracked_objects)}, "
                f"MotionPx=(dx={msg.estimated_motion_dx:.1f}, dy={msg.estimated_motion_dy:.1f})"
            )
        if self.motion_model:
            self.motion_model.update_from_tracking(
                msg.estimated_motion_dx, msg.estimated_motion_dy, msg_time
            )
        if self.object_manager:
            self.object_manager.update_from_tracking(msg.tracked_objects, msg.header)

    def _spraying_state_callback(self, msg: StringMsg):
        """
        Handles messages from the spraying_state_control node to change strategy.
        """
        state_command = msg.data
        self.get_logger().info(f"Received spraying state command: '{state_command}'")

        target_strategy_type: Optional[str] = None

        if state_command == "":
            target_strategy_type = self._default_strategy_type
            if target_strategy_type is None:
                self.get_logger().error("Cannot revert to default strategy: Default type unknown.")
                return
            self.get_logger().info(f"Command is empty string, reverting to default strategy: '{target_strategy_type}'")
        elif state_command == "NA":
            target_strategy_type = "no_spray"
        elif state_command == "PA":
            target_strategy_type = "spray_all"
        elif state_command == "PAS":
            target_strategy_type = "simple_weed"
        else:
            self.get_logger().warn(f"Unknown spraying state command: '{state_command}'. No strategy change.")
            return

        if target_strategy_type:
            if self.strategy and self.strategy.__class__.__name__.lower().replace("strategy","") == target_strategy_type.lower().replace("_", ""):
                self.get_logger().info(f"Requested strategy '{target_strategy_type}' is already active. No change.")
            else:
                self.request_strategy_change(target_strategy_type)
        else:
            self.get_logger().error(f"Could not determine target strategy for command '{state_command}'.")


    def _setup_control_loop(self):
        loop_frequency = self.config.get('control_loop_frequency', 1.0)
        if loop_frequency <= 0:
            self.get_logger().error(f"Invalid control_loop_frequency ({loop_frequency}). Disabling loop.")
            return
        timer_period = 1.0 / loop_frequency
        self.control_timer = self.create_timer(timer_period, self._timer_callback)
        self.get_logger().info(f"Control loop started: {loop_frequency:.1f} Hz (Period: {timer_period*1000:.2f} ms)")

    def _timer_callback(self):
        loop_start_mono = time.monotonic()

        display_states: List[DisplayObjectState] = []
        all_managed_objects_snapshot: List[ManagedObject] = []
        current_nozzle_command: List[bool] = []
        current_velocity_cmps: int = 0

        # 1. Update object predictions, their bounding boxes (with safety), and get display state
        if self.object_manager:
            display_states = self.object_manager.update_predictions_and_get_display_state()
            all_managed_objects_snapshot = self.object_manager.get_all_managed_objects()

        # 2. Get current velocity
        current_velocity_mps = (0.0, 0.0)
        if self.motion_model:
            current_velocity_mps = self.motion_model.get_velocity_mps()
            _, vy_mps = current_velocity_mps
            velocity_cmps_float = vy_mps * 100.0
            current_velocity_cmps = max(0, min(int(round(velocity_cmps_float)), SerialRelayInterface.MAX_VELOCITY_PAYLOAD))

        # 3. Map Objects to Nozzles (for visualization assignment, not primary scheduling)
        objects_for_mapping = [obj for obj in all_managed_objects_snapshot if obj.status in [ObjectStatus.TARGETED, ObjectStatus.SPRAYING, ObjectStatus.SCHEDULED]]
        target_to_nozzle_map: Dict[int, List[int]] = {}
        if self.nozzle_mapper and objects_for_mapping:
            target_to_nozzle_map = self.nozzle_mapper.map_objects_to_nozzles(objects_for_mapping)

        # 4. Determine current nozzle commands using SprayScheduler
        num_hw_nozzles = self.nozzle_config.num_nozzles if self.nozzle_config else 0
        if self.spray_scheduler:
            current_nozzle_command = self.spray_scheduler.get_current_nozzle_state(
                all_managed_objects_snapshot, # Pass all objects
                current_velocity_mps
            )
        else:
            effective_nozzles = min(num_hw_nozzles, SerialRelayInterface.MAX_SUPPORTED_NOZZLES)
            current_nozzle_command = [False] * effective_nozzles

        if 0 < len(current_nozzle_command) < num_hw_nozzles <= SerialRelayInterface.MAX_SUPPORTED_NOZZLES:
             current_nozzle_command.extend([False] * (num_hw_nozzles - len(current_nozzle_command)))

        # 5. Update debug GUI
        if self.debug_gui:
            # Pass all_managed_objects_snapshot to GUI so it can display statuses updated by scheduler
            self.debug_gui.update_display_data(
                display_states=display_states, # display_states is for drawing, might not have latest status post-scheduling
                scheduled_objects=all_managed_objects_snapshot, # This list has objects with statuses updated by scheduler
                nozzle_config=self.nozzle_config,
                target_nozzle_map=target_to_nozzle_map,
                current_nozzle_command=current_nozzle_command
            )

        # 6. Hardware control
        if self.hardware_interface and self.hardware_interface.is_connected():
            self.hardware_interface.set_nozzle_state(current_nozzle_command, current_velocity_cmps)
        elif any(current_nozzle_command):
            log_method = self.get_logger().error if not (self.hardware_interface and self.hardware_interface.is_connected()) else self.get_logger().warning
            log_method("Spray command generated, but hardware not available/connected.", throttle_duration_sec=5)

        # 7. Loop timining check
        loop_end_mono = time.monotonic()
        duration_ms = (loop_end_mono - loop_start_mono) * 1000.0
        target_period_ms = 1000.0 / self.config.get('control_loop_frequency', 1.0)
        if duration_ms > target_period_ms * 1.10:
            self.get_logger().warning(
                f"Loop duration ({duration_ms:.2f}ms) > target ({target_period_ms:.2f}ms).",
                throttle_duration_sec=1.0
            )

    def request_strategy_change(self, new_strategy_type: str):
        self.get_logger().info(f"Request to switch strategy to: '{new_strategy_type}'")
        current_full_strategy_config = self.config.get('strategy', {})
        new_strategy_instance = create_spraying_strategy(
            strategy_config=current_full_strategy_config,
            logger=self.get_logger(), clock=self.get_clock(),
            strategy_type_override=new_strategy_type
        )

        if new_strategy_instance is None:
            self.get_logger().error(f"Failed to create strategy '{new_strategy_type}'. No change.")
            return

        with self._strategy_update_lock:
            if self.object_manager is None:
                self.get_logger().error("ObjectManager not available for strategy change.")
                return

            old_strategy_name = self.strategy.__class__.__name__ if self.strategy else "None"
            self.strategy = new_strategy_instance
            self.object_manager.set_strategy(self.strategy)

            if self.debug_gui:
                self.debug_gui.update_active_strategy_name(new_strategy_type)
            self.get_logger().info(f"Switched strategy from {old_strategy_name} to {self.strategy.__class__.__name__}")

    def on_shutdown(self):
        self.get_logger().info(f"Shutting down {self.get_name()}...")
        if self.debug_gui: self.debug_gui.stop()
        if self.control_timer is not None and not self.control_timer.canceled:
            self.control_timer.cancel()
        if self.hardware_interface: self.hardware_interface.disconnect()
        self.get_logger().info(f"{self.get_name()} shutdown complete.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 0
    try:
        node = SprayerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger = node.get_logger() if node else logging.getLogger("sprayer_node.main")
        logger.info('Keyboard interrupt, shutting down.')
    except SystemExit as e:
        logger = node.get_logger() if node else logging.getLogger("sprayer_node.main")
        logger.info(f"SystemExit: {e}")
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception:
        logger = node.get_logger() if node else logging.getLogger("sprayer_node.main")
        logger.fatal(f"Unhandled exception in SprayerNode:\n{traceback.format_exc()}")
        exit_code = 1
    finally:
        if node:
            node.on_shutdown()
            if rclpy.ok():
                if node.context and node.context.ok():
                    try:
                        node.destroy_node()
                    except rclpy.exceptions.InvalidHandle:
                        node.get_logger().warn("Node handle invalid during destroy_node (possibly already destroyed).")
                    except Exception as destroy_e:
                        node.get_logger().error(f"Error destroying node: {destroy_e}")
                else:
                    print("SprayerNode: ROS context invalid before destroy_node, skipping node destruction.")
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()