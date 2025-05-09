import logging
from typing import Dict, Any, Set, List
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from ..common.definitions import ManagedObject

class TemplateStrategy(BaseSprayStrategy):
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        super().__init__(strategy_config, logger, clock)
        self._logger.info(f"Initializing {self.__class__.__name__}...")
        pass

    def decide(self, target: ManagedObject) -> bool:
        # DEFINE STRATEGY FOR SINGLE OBJECT HERE
        # see "ManagedObject" in common/definitions.py for available info to define strat
        pass

    def get_min_target_coverage_ratio(self) -> float:
        return self._min_coverage

    def get_max_nontarget_overspray_ratio(self) -> float:
        return self._max_overspray