import logging
from typing import Dict, Any, Set, List
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from ..common.definitions import ManagedObject

class SimpleWeedStrategy(BaseSprayStrategy):
    """
    Simple weed spraying strategy.

    Targets objects if:
    1. Confidence >= min_confidence
    2. Class name is NOT in the configured crop_class_names list.
    Applies a safety zone around configured crop_class_names.
    """
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        super().__init__(strategy_config, logger, clock)
        self._logger.info(f"Initializing {self.__class__.__name__}...")

        try:
            crop_names_list: List[str] = self._config.get('crop_class_names', [])
            if not isinstance(crop_names_list, list):
                raise ValueError("'crop_class_names' must be a list of strings.")
            self._crop_class_names: Set[str] = set(crop_names_list)

            self._min_confidence: float = float(self._config.get('min_confidence', 0.4))
            if not (0.0 <= self._min_confidence <= 1.0):
                raise ValueError("min_confidence must be between 0.0 and 1.0")

            self._safety_zone_cm: float = float(self._config.get('safety_zone_in_cm', 0.0))
            if self._safety_zone_cm < 0.0:
                self._logger.warning(f"safety_zone_in_cm ({self._safety_zone_cm}cm) is negative. Clamping to 0.0cm.")
                self._safety_zone_cm = 0.0
            self._safety_zone_m: float = self._safety_zone_cm / 100.0

            self._parse_common_constraints() 

        except (ValueError, TypeError) as e:
            self._logger.error(f"Configuration error for {self.__class__.__name__}: {e}")
            raise ValueError(f"Configuration error in {self.__class__.__name__}") from e

        if not self._crop_class_names:
            self._logger.warning("No 'crop_class_names' configured for SimpleWeedStrategy. All non-low-confidence objects might be targeted.")
        else:
            self._logger.info(f"  Crop Classes (Protected): {sorted(list(self._crop_class_names))}")
            self._logger.info(f"  Safety Zone for Protected Classes: {self._safety_zone_cm:.1f} cm ({self._safety_zone_m:.3f} m)")
        self._logger.info(f"  Min Confidence Threshold: {self._min_confidence:.2f}")


    def decide(self, target: ManagedObject) -> bool:
        if target.confidence < self._min_confidence:
            return False 

        # Avoid min size in meters
        obj_width_m, obj_length_m = target.size_m
        if obj_width_m <= 0.005 and obj_length_m <= 0.005: # 5mm
            return False

        if target.class_name in self._crop_class_names:
            return False # Do not spray crops

        return True # Spray if it's not a crop and meets confidence/size

    def get_safety_zone_m(self, target: ManagedObject) -> float:
        if target.class_name in self._crop_class_names:
            return self._safety_zone_m
        return 0.0

    def get_min_target_coverage_ratio(self) -> float:
        return self._min_coverage

    def get_max_nontarget_overspray_ratio(self) -> float:
        return self._max_overspray