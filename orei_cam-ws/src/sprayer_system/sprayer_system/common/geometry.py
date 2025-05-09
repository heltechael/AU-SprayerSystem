import logging
from typing import Tuple
from .definitions import CameraParams

logger = logging.getLogger(__name__)

# IMPORTANT: functions use following assumptions:
#   1. The camera looks straight down (zero pitch/roll)
#   2. The provided GSD is constant across the image
#   3. The robot's ground frame origin (0,0) corresponds to the
#      center of the image projected onto the ground

def image_px_to_robot_ground_m(
    pixel_coords: Tuple[float, float],
    cam_params: CameraParams
) -> Tuple[float, float]:
    px_x, px_y = pixel_coords

    px_rel_center_x = px_x - cam_params.center_x_px
    px_rel_center_y = px_y - cam_params.center_y_px

    ground_x_m = px_rel_center_x * cam_params.m_per_px
    ground_y_m = px_rel_center_y * cam_params.m_per_px

    return ground_x_m, ground_y_m

def robot_ground_m_to_image_px(
    ground_coords_m: Tuple[float, float],
    cam_params: CameraParams
) -> Tuple[float, float]:
    ground_x_m, ground_y_m = ground_coords_m

    px_rel_center_x = ground_x_m / cam_params.m_per_px
    px_rel_center_y = ground_y_m / cam_params.m_per_px 

    px_x = px_rel_center_x + cam_params.center_x_px
    px_y = px_rel_center_y + cam_params.center_y_px

    return px_x, px_y


def estimate_object_size_m(
    bbox_image_px: Tuple[float, float, float, float], # x,y,w,h
    cam_params: CameraParams
) -> Tuple[float, float]:
    _, _, width_px, height_px = bbox_image_px
    # Simple scaling using average GSD
    width_m = width_px * cam_params.m_per_px
    length_m = height_px * cam_params.m_per_px 
    return width_m, length_m


def calibration_px_to_robot_ground_m(
    calib_pixel_coords: Tuple[int, int],
    node_cam_params: CameraParams 
) -> Tuple[float, float]:
    return image_px_to_robot_ground_m(calib_pixel_coords, node_cam_params)