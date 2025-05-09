import logging
from typing import Dict, Any, Set, List
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from ..common.definitions import ManagedObject

SPRAY_SPECIES = {"TAROF", "CHEAL", "EQUAR", "1MATG",
                 "GALAP", "SINAR", "1CRUF", "CIRAR", "POLCO"}
PRESERVE_SPECIES = {"PIBSA", "VIOAR", "GERMO", "EPHHE", "LAMPU"}
CHECK_SIZE_SPECIES = {"FUMOF", "POLLA", "POLAV", "ATXPA", "VERPE"}
CONFIDENCE = 0.4

class MarcoStrategy(BaseSprayStrategy):
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        super().__init__(strategy_config, logger, clock)
        self._logger.info(f"Initializing {self.__class__.__name__}...")
        self.spray_list = SPRAY_SPECIES
        self.preserve_list = PRESERVE_SPECIES
        self.check_size_list = CHECK_SIZE_SPECIES
        self.confidence = CONFIDENCE

    def decide(self, target: ManagedObject) -> bool:
        species = target.species
        size = target.bounding_box_size
        confidence = target.confidence

        # below threshold confidence, do not spray
        if confidence < self.confidence:
            return False

        if species in self.spray_list:
            return True
        elif species in self.preserve_list:
            return False
        elif species in self.check_size_list:
            # Check the size condition
            if size > 40:
                return True
            else:
                return False
        else:
            # For unknown species, check the size condition
            if size > 40:
                return True
            else:
                return False

    def get_min_target_coverage_ratio(self) -> float:
        return self._min_coverage

    def get_max_nontarget_overspray_ratio(self) -> float:
        return self._max_overspray