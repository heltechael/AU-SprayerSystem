import logging
from typing import Dict, Any
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from ..common.definitions import ManagedObject

class SprayAllStrategy(BaseSprayStrategy):
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        super().__init__(strategy_config, logger, clock)
        self._logger.info(f"Initializing {self.__class__.__name__}...")

        try:
            self._min_confidence: float = float(self._config.get('min_confidence', 0.1)) 
            if not (0.0 <= self._min_confidence <= 1.0):
                raise ValueError("min_confidence must be between 0.0 and 1.0")

            self._parse_common_constraints() 

        except (ValueError, TypeError) as e:
            self._logger.error(f"Configuration error for {self.__class__.__name__}: {e}")
            raise ValueError(f"Configuration error in {self.__class__.__name__}") from e

        self._logger.info(f"  Min Confidence Threshold: {self._min_confidence:.2f}")


    def decide(self, target: ManagedObject) -> bool:
        if target.confidence < self._min_confidence:
            return False
        return True

    def get_safety_zone_m(self, target: ManagedObject) -> float:
        return 0.0

    def get_min_target_coverage_ratio(self) -> float:
        return self._min_coverage

    def get_max_nontarget_overspray_ratio(self) -> float:
        return self._max_overspray