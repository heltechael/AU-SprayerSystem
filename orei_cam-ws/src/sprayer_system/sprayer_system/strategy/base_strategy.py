from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from rclpy.clock import Clock
from ..common.definitions import ManagedObject

class BaseSprayStrategy(ABC):
    """
    Abstract Base Class defining the interface for spray decision strategies
    """
    def __init__(self, strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock):
        self._config = strategy_config
        self._logger = logger
        self._clock = clock
        self._min_coverage = 0.9 # default
        self._max_overspray = 0.5 # default

    @abstractmethod
    def decide(self, target: ManagedObject) -> bool:
        """
        Decides if the given target object should be sprayed

        Args:
            target: The ManagedObject instance to evaluate

        Returns:
            True if the object should be targeted for spraying, False otherwise
        """
        pass

    @abstractmethod
    def get_min_target_coverage_ratio(self) -> float:
        """
        Returns the minimum required coverage ratio (0.0-1.0) for objects that this strategy decides should be sprayed. Used by planners
        """
        pass

    @abstractmethod
    def get_max_nontarget_overspray_ratio(self) -> float:
        """
        Returns the maximum allowable overspray ratio (0.0-1.0) onto objects that this strategy decides should *not* be sprayed. Used by planners
        """
        pass

    @abstractmethod
    def get_safety_zone_m(self, target: ManagedObject) -> float:
        """
        Returns the safety zone margin in meters for the given target.
        A non-zero value indicates the object (and this margin around it) should be protected.
        """
        pass

    def _parse_common_constraints(self):
        """
        Helper method to parse and validate common constraint parameters from the strategy's configuration dictionary. Concrete strategies should call this
        """
        try:
            self._min_coverage = float(self._config.get('min_target_coverage_ratio', 0.9))
            self._max_overspray = float(self._config.get('max_nontarget_overspray_ratio', 0.1))

            if not (0.0 <= self._min_coverage <= 1.0):
                raise ValueError("min_target_coverage_ratio must be between 0.0 and 1.0")
            if not (0.0 <= self._max_overspray <= 1.0):
                raise ValueError("max_nontarget_overspray_ratio must be between 0.0 and 1.0")

            self._logger.info(f"  Strategy Constraints Parsed: Min Coverage={self._min_coverage:.2f}, Max Overspray={self._max_overspray:.2f}")

        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid constraint configuration: {e}") from e