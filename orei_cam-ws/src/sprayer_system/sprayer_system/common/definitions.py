from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto
import time
import numpy as np

from rclpy.time import Time

class ObjectStatus(Enum):
    """Represents the state of a potential spray object."""
    PENDING = auto()        # Newly identified or re-evaluating strategy decision
    TARGETED = auto()       # Strategy decided YES, pending planning/scheduling
    IGNORED = auto()        # Strategy decided NO
    SCHEDULED = auto()      # Spray timing/nozzles determined by scheduler
    SPRAYING = auto()       # Actively being sprayed (within calculated window + latency effects)
    SPRAYED = auto()        # Has passed through the spray window and duration
    MISSED = auto()         # Passed through window but wasn't sprayed (e.g., error, disabled)
    LOST = auto()           # Track lost by vision system before completion or spray
    APPROACHING = auto()    # Optional: If planner wants to mark objects nearing boom


@dataclass
class Nozzle:
    """
    Represents single nozzle on the spray boom
    Includes its physical position, its calibrated spray pattern
    and its specific activation Y-coordinate for scheduling
    """
    index: int
    position_m: Tuple[float, float] # Absolute position on the boom (X, Y) in robot frame
    # Calibrated spray pattern polygon vertices, relative to position_m (meters)
    spray_pattern_relative_m: List[Tuple[float, float]] = field(default_factory=list)
    # Bounding box of the relative spray pattern (rel_min_x, rel_min_y, rel_max_x, rel_max_y)
    bounding_box_relative_m: Optional[Tuple[float, float, float, float]] = field(default=None, init=False)
    # Absolute Y-coordinate (robot ground frame) of this nozzle's activation line
    # Derived from the center of its calibrated spray pattern
    activation_y_m: Optional[float] = None

    def __post_init__(self):
        """Calculates the relative bounding box after initialization or pattern update"""
        self.recalculate_bounding_box()

    def recalculate_bounding_box(self):
        """Helper to calculate the bounding box from the relative pattern"""
        if self.spray_pattern_relative_m:
            points = np.array(self.spray_pattern_relative_m)
            if points.size > 0:
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                self.bounding_box_relative_m = (
                    float(min_coords[0]), float(min_coords[1]),
                    float(max_coords[0]), float(max_coords[1])
                )
            else:
                self.bounding_box_relative_m = None
        else:
            self.bounding_box_relative_m = None

@dataclass
class CameraParams:
    """Holds necessary camera parameters for coordinate transformations"""
    width_px: int
    height_px: int
    gsd_px_per_m: float
    m_per_px: float = field(init=False)
    center_x_px: float = field(init=False)
    center_y_px: float = field(init=False)

    def __post_init__(self):
        if self.gsd_px_per_m <= 0: raise ValueError("GSD must be positive")
        self.m_per_px = 1.0 / self.gsd_px_per_m
        self.center_x_px = self.width_px / 2.0
        self.center_y_px = self.height_px / 2.0

@dataclass
class ManagedObject:
    """Holds the state and information for an object being considered for spraying"""
    track_id: int
    class_id: int
    class_name: str
    last_seen_time: Time
    bbox_image: Tuple[float, float, float, float] # Original x,y,w,h in pixels from vision
    confidence: float
    last_vision_class_name: str = ""
    last_vision_confidence: float = 0.0
    position_robot_m: Tuple[float, float] = (0.0, 0.0) # Absolute center (X, Y) based on bbox_image center
    predicted_position_robot_m: Tuple[float, float] = (0.0, 0.0) # Continuously updated absolute center (X, Y)
    size_m: Tuple[float, float] = (0.0, 0.0) # Original (width_m, length_m) based on bbox_image w,h
    status: ObjectStatus = ObjectStatus.PENDING
    assigned_nozzle_indices: List[int] = field(default_factory=list)
    spray_start_time: Optional[Time] = None
    spray_end_time: Optional[Time] = None
    created_time: Time = field(default_factory=Time)
    
    bounding_box_m: Optional[Tuple[float, float, float, float]] = field(default=None, init=False)
    activation_bounding_box_m: Optional[Tuple[float, float, float, float]] = field(default=None, init=False)
    safety_bounding_box_m: Optional[Tuple[float, float, float, float]] = field(default=None, init=False)
    current_safety_margin_m: float = 0.0


    def update_bounding_boxes(self,
                              cam_params: CameraParams,
                              config_object_activation_width_px: Optional[float] = None,
                              safety_margin_m: float = 0.0):
        # 1. Update self.bounding_box_m (original object size at predicted position)
        pred_x, pred_y = self.predicted_position_robot_m
        width_m_orig, length_m_orig = self.size_m
        
        half_w_orig = width_m_orig / 2.0
        half_l_orig = length_m_orig / 2.0
        self.bounding_box_m = (
            pred_x - half_w_orig, pred_y - half_l_orig,
            pred_x + half_w_orig, pred_y + half_l_orig
        )

        # 2. Update self.activation_bounding_box_m (potentially shrunk width for activation)
        original_pixel_width = self.bbox_image[2] 
        activation_pixel_width = original_pixel_width

        if config_object_activation_width_px is not None and \
           0 < config_object_activation_width_px < original_pixel_width:
            activation_pixel_width = config_object_activation_width_px
        
        activation_width_m = activation_pixel_width * cam_params.m_per_px
        activation_length_m = length_m_orig 

        half_act_w = activation_width_m / 2.0
        half_act_l = activation_length_m / 2.0 

        self.activation_bounding_box_m = (
            pred_x - half_act_w, pred_y - half_act_l, 
            pred_x + half_act_w, pred_y + half_act_l
        )

        # 3. Update self.safety_bounding_box_m
        self.current_safety_margin_m = safety_margin_m
        if safety_margin_m > 0.0001 and self.bounding_box_m:
            orig_min_x, orig_min_y, orig_max_x, orig_max_y = self.bounding_box_m
            self.safety_bounding_box_m = (
                orig_min_x - safety_margin_m,
                orig_min_y - safety_margin_m,
                orig_max_x + safety_margin_m,
                orig_max_y + safety_margin_m,
            )
        else:
            self.safety_bounding_box_m = None


@dataclass
class DisplayObjectState:
    """Data prepared by ObjectManager for the debuggui"""
    track_id: int
    predicted_center_image_px: Tuple[float, float]
    original_bbox_dims_px: Tuple[float, float] # Original width, height in GUI scaled pixels
    activation_bbox_dims_px: Tuple[float, float] # Activation width, height in GUI scaled pixels
    label: str
    status: ObjectStatus
    safety_bbox_gui_px: Optional[Tuple[float, float, float, float]] = None # (min_x, min_y, max_x, max_y)