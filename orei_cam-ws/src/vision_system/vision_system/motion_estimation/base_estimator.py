from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np

class BaseMotionEstimator(ABC):
    """Abstract Base Class for motion estimation algorithms"""
    def __init__(self, config: Dict[str, Any], logger):
        self.logger = logger
        self._config = config
        self.logger.info(f"[{self.__class__.__name__}] Initializing...")

    @abstractmethod
    def estimate_displacement(self, current_frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        pass

    @abstractmethod
    def reset_state(self) -> None:
        pass