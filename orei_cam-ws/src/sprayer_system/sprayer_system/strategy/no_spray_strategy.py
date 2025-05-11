import logging
from typing import Dict, Any
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from ..common.definitions import ManagedObject

class NoSprayStrategy(BaseSprayStrategy):
    """
    A strategy that does not spray any object
    """
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        super().__init__(strategy_config, logger, clock)
        self._logger.info(f"Initializing {self.__class__.__name__}...")
        self._parse_common_constraints()
        self._logger.info(f"  {self.__class__.__name__}: All objects will be IGNORED.")

    def decide(self, target: ManagedObject) -> bool:
        # never spray any object
        return False

    def get_safety_zone_m(self, target: ManagedObject) -> float:
        return 0.0

    def get_min_target_coverage_ratio(self) -> float:
        return self._min_coverage

    def get_max_nontarget_overspray_ratio(self) -> float:
        return self._max_overspray