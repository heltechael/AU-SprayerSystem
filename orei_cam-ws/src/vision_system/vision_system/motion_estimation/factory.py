from typing import Dict, Any, Optional

from .base_estimator import BaseMotionEstimator
from .orb_homography_estimator import OrbHomographyEstimator
from .sparse_optical_flow_estimator import SparseOpticalFlowEstimator

def create_motion_estimator(estimator_type: str, config: Dict[str, Any], logger) -> Optional[BaseMotionEstimator]:
    logger.info(f"[MotionEstimatorFactory] Attempting to create estimator of type: '{estimator_type}'")

    if estimator_type == "orb_homography":
        return OrbHomographyEstimator(config, logger)
    elif estimator_type == "sparse_lk":
        return SparseOpticalFlowEstimator(config, logger)
    # we can add more branches here for other motion estimator methods if needed
    else:
        logger.error(f"[MotionEstimatorFactory] Unknown estimator type: '{estimator_type}'")
        return None