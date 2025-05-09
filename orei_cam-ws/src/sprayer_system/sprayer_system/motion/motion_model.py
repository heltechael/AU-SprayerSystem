import logging
from typing import Optional, Tuple, Dict, Any

from rclpy.time import Time, Duration
from rclpy.clock import Clock
from gps_msgs.msg import GPSFix, GPSStatus 
from ..common.definitions import CameraParams

class MotionModel:
    """
    Estimates robot ground velocity (m/s) relative to the robot frame (+Y forward, +X right)
    Uses valid, non-stale GPS speed for the primary forward component (vy)
    Uses vision-derived pixel displacement (dx, dy) scaled by GSD for the lateral component (vx)
    and as a fallback for the forward component (vy) if GPS is invalid or stale
    Applies playback speed factor directly (for ROS BAG playbacks at lower speeds)
    """
    MIN_VALID_GPS_STATUS = GPSStatus.STATUS_FIX 

    def __init__(self,
                 config: Dict[str, Any],
                 cam_params: CameraParams,
                 logger: logging.Logger,
                 clock: Clock):
        self._config = config
        self._cam_params = cam_params
        self._logger = logger
        self._clock = clock
        self._last_gps_time: Optional[Time] = None
        self._latest_gps_speed_mps: Optional[float] = None 
        self._last_gps_is_valid: bool = False
        self._gps_staleness_threshold_sec: float = float(self._config.get('gps_staleness_threshold_sec', 1.0))
        self._last_vision_update_time: Optional[Time] = None
        self._raw_vx_mps_vision: float = 0.0
        self._raw_vy_mps_vision: float = 0.0
        self._output_vx_mps: float = 0.0
        self._output_vy_mps: float = 0.0
        self._vy_source: str = "Startup"
        self._playback_speed_factor: float = float(self._config.get('playback_speed_factor', 1.0))

        log_suffix = ""
        if self._playback_speed_factor != 1.0:
             log_suffix = f" (Playback Speed Factor: {self._playback_speed_factor:.2f})"
        self._logger.info(f"MotionModel initialized (m/s){log_suffix}")
        if self._gps_staleness_threshold_sec > 0:
            self._logger.info(f"  GPS Staleness Check: Enabled (threshold={self._gps_staleness_threshold_sec:.2f}s)")
        else:
            self._logger.info("  GPS Staleness Check: Disabled")
        self._logger.info(f"  Minimum Valid GPS Status: {self.MIN_VALID_GPS_STATUS}")


    def update_from_tracking(self,
                             estimated_dx_px: float,
                             estimated_dy_px: float,
                             message_time: Time):
        """
        Updates the vision-derived velocity components (vx, vy) in m/s
        These are used for the lateral component (vx) and as a fallback for vy
        """
        if message_time.nanoseconds == 0:
            self._logger.warn("MotionModel (Vision Update): Received zero timestamp. Cannot calculate velocity.")
            return

        if self._last_vision_update_time is None:
            self._logger.debug(f"MotionModel (Vision Update): Storing first vision timestamp: {message_time.nanoseconds / 1e9:.3f}s.")
            self._last_vision_update_time = message_time
            return

        dt_duration: Duration = message_time - self._last_vision_update_time
        dt_sec = dt_duration.nanoseconds / 1e9

        #self._logger.info(f"[MOTION DT DEBUG] Current Msg Time: {message_time.nanoseconds / 1e9:.6f}")
        if self._last_vision_update_time:
            #self._logger.info(f"[MOTION DT DEBUG] Last Vision Time: {self._last_vision_update_time.nanoseconds / 1e9:.6f}")
            dt_duration: Duration = message_time - self._last_vision_update_time
            dt_sec = dt_duration.nanoseconds / 1e9
            #self._logger.info(f"[MOTION DT DEBUG] Calculated dt_sec: {dt_sec:.6f} | dx_px: {estimated_dx_px:.2f}, dy_px: {estimated_dy_px:.2f}")
            if dt_sec > 1e-9:
                try: 
                    dx_m = estimated_dx_px * self._cam_params.m_per_px
                    dy_m = estimated_dy_px * self._cam_params.m_per_px
                    raw_vx = dx_m / dt_sec
                    raw_vy = dy_m / dt_sec
                    #self._logger.info(f"[MOTION DT DEBUG] Raw Vision Vel (m/s): vx={raw_vx:.3f}, vy={raw_vy:.3f}")
                except Exception as calc_e:
                    self._logger.error(f"[MOTION DT DEBUG] Error in debug velocity calc: {calc_e}")
            else:
                self._logger.warning("[MOTION DT DEBUG] dt_sec too small for velocity calculation.")
        else:
            self._logger.info("[MOTION DT DEBUG] First vision message, skipping dt calculation")

        # sanity dt check
        max_reasonable_dt = 5.0
        if dt_sec <= 1e-9:
            # self._logger.warn(f"MotionModel (Vision Update): Non-positive vision dt ({dt_sec:.4f}s). Resetting vision velocity.")
            self._last_vision_update_time = message_time
            self._raw_vx_mps_vision = 0.0
            self._raw_vy_mps_vision = 0.0
            self._update_output_velocity()
            return
        elif dt_sec > max_reasonable_dt:
             self._logger.warn(f"MotionModel (Vision Update): Unusually large vision dt ({dt_sec:.4f}s > {max_reasonable_dt}s). Resetting vision velocity.")
             self._raw_vx_mps_vision = 0.0
             self._raw_vy_mps_vision = 0.0
             self._last_vision_update_time = message_time 
             self._update_output_velocity()
             return

        # convert pixel displacement to meters using GSD 
        try:
            dx_m = estimated_dx_px * self._cam_params.m_per_px
            dy_m = estimated_dy_px * self._cam_params.m_per_px
        except Exception as e:
             self._logger.error(f"MotionModel (Vision Update): Error converting pixels to meters: {e}")
             dx_m = 0.0
             dy_m = 0.0

        # calculate vision-based velocity (m/s) 
        try:
            self._raw_vx_mps_vision = dx_m / dt_sec
            self._raw_vy_mps_vision = dy_m / dt_sec
        except ZeroDivisionError:
             self._logger.warn(f"MotionModel (Vision Update): Zero division error calculating vision velocity (dt={dt_sec:.4f}s).")
             self._raw_vx_mps_vision = 0.0
             self._raw_vy_mps_vision = 0.0
        except Exception as e:
            self._logger.error(f"MotionModel (Vision Update): Error calculating vision velocity: {e}")
            self._raw_vx_mps_vision = 0.0
            self._raw_vy_mps_vision = 0.0

        self._update_output_velocity() # 
        self._last_vision_update_time = message_time


    def update_from_gps(self, gps_msg: GPSFix):
        """
        Updates the internal state with the latest GPS speed and timestamp - checking for validity
        """
        current_gps_time = Time.from_msg(gps_msg.header.stamp)
        current_gps_status = gps_msg.status.status
        current_gps_unix_time = gps_msg.time 

        is_currently_valid = (
            current_gps_time.nanoseconds > 0 and
            current_gps_unix_time > 0 and 
            current_gps_status >= self.MIN_VALID_GPS_STATUS and
            gps_msg.speed == gps_msg.speed 
        )

        self._last_gps_time = current_gps_time
        self._last_gps_is_valid = is_currently_valid 

        if is_currently_valid:
            if gps_msg.speed >= 0:
                self._latest_gps_speed_mps = gps_msg.speed
                # self._logger.debug(f"MotionModel (GPS Update): Valid GPS received. Speed: {self._latest_gps_speed_mps:.2f} m/s")
            else:
                # This case might be unlikely if status check passes - fine to keep
                self._logger.warn(f"MotionModel (GPS Update): Valid GPS status but negative speed ({gps_msg.speed:.2f} m/s). Treating as invalid speed.")
                self._latest_gps_speed_mps = None 
                self._last_gps_is_valid = False 
        else:
            if self._latest_gps_speed_mps is not None: 
                 self._logger.warn(f"MotionModel (GPS Update): Received invalid GPS data (Status: {current_gps_status}, Time: {current_gps_unix_time:.2f}, HeaderStamp: {current_gps_time.nanoseconds}). Clearing GPS speed.")
            self._latest_gps_speed_mps = None

        self._update_output_velocity() 


    def _update_output_velocity(self):
        """
        Selects appropriate velocity components (GPS-priority for vy if valid and not stale)
        and applies the playback speed factor directly. Records the source of vy
        """
        current_vx = self._raw_vx_mps_vision
        current_vy = 0.0
        vy_source = "Unknown" 

        # should gps be used for vy?
        is_stale = self.is_gps_data_stale()
        use_gps_for_vy = self._last_gps_is_valid and not is_stale and self._latest_gps_speed_mps is not None

        if use_gps_for_vy:
            current_vy = self._latest_gps_speed_mps
            vy_source = "GPS"
        else:
            # use vision vy as fallback
            current_vy = self._raw_vy_mps_vision
            vy_source = "VisionFallback"
            log_reason = ""
            if not self._last_gps_is_valid: log_reason += "Invalid "
            if is_stale: log_reason += "Stale "
            if self._latest_gps_speed_mps is None and self._last_gps_is_valid and not is_stale: log_reason += "NoSpeed " # probably rare
            if not log_reason: log_reason = "UnknownReason" # Fallback reason

            self._logger.warn(f"MotionModel: Using VisionFallback for Vy. Reason: {log_reason.strip()}", throttle_duration_sec=2.0)

        self._output_vx_mps = current_vx * self._playback_speed_factor
        self._output_vy_mps = current_vy * self._playback_speed_factor
        self._vy_source = vy_source 

        # THIS WILL MAKE COMPUTATION EXCEED 128 HZ DEADLINES!!! ONLY LOG FOR VELOCITY RELATED DEBUGGING
        # self._logger.debug(f"MotionModel Updated Output Velocity: Vx={self._output_vx_mps:.2f}, Vy={self._output_vy_mps:.2f} m/s (Vy Source: {self._vy_source})")

    def get_velocity_mps(self) -> Tuple[float, float]:
        return self._output_vx_mps, self._output_vy_mps

    def get_vy_source(self) -> str:
        return self._vy_source

    def is_gps_data_stale(self) -> bool:
        if self._gps_staleness_threshold_sec <= 0:
            return False 

        if self._last_gps_time is None:
            return True 

        now = self._clock.now()
        age_duration = now - self._last_gps_time
        age_sec = age_duration.nanoseconds / 1e9

        is_stale = age_sec > self._gps_staleness_threshold_sec
        if is_stale:
            self._logger.debug(f"GPS data is stale (Age: {age_sec:.2f}s > Threshold: {self._gps_staleness_threshold_sec:.2f}s)")
        return is_stale

    def get_last_gps_time(self) -> Optional[Time]:
        return self._last_gps_time