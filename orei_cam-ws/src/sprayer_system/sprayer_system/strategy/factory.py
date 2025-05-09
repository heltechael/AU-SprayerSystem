from typing import Dict, Any, Optional
import logging
import traceback
from rclpy.clock import Clock
from .base_strategy import BaseSprayStrategy
from .simple_weed_strategy import SimpleWeedStrategy
from .spray_all_strategy import SprayAllStrategy 
from .macro_strategy import MarcoStrategy 

# Import other strategies here, e.g. MarcoStrat{2/3/4}

def create_spraying_strategy(strategy_config: Dict[str, Any], logger: logging.Logger, clock: Clock, strategy_type_override: Optional[str] = None) -> Optional[BaseSprayStrategy]:
    strategy_type = strategy_type_override if strategy_type_override is not None else strategy_config.get('type')

    logger.info(f"Attempting to create spraying strategy of type: '{strategy_type}'")

    if not strategy_type:
        logger.error("Strategy 'type' not specified in configuration or override.")
        return None

    strategy_type_lower = strategy_type.lower()

    try:
        if strategy_type_lower == "simple_weed":
            config_for_simple = strategy_config.get('simple_weed', {})
            if not config_for_simple: logger.warning(f"No config under 'strategy.simple_weed'.")
            return SimpleWeedStrategy(config_for_simple, logger, clock)

        elif strategy_type_lower == "spray_all": 
            config_for_spray_all = strategy_config.get('spray_all', {})
            if not config_for_spray_all: logger.warning(f"No config under 'strategy.spray_all'.")
            return SprayAllStrategy(config_for_spray_all, logger, clock)
        
        elif strategy_type_lower == "macro_strategy": 
            config_for_spray_all = strategy_config.get('macro_strategy', {})
            if not config_for_spray_all: logger.warning(f"No config under 'strategy.macro_strategy'.")
            return SprayAllStrategy(config_for_spray_all, logger, clock)

        # --- Add other strategies here ---
        # elif strategy_type_lower == "marco_strat1":
        #     config_for_macro_strat1 = strategy_config.get('marco_strat1', {})
        #     ...
        # ---------------------------------

        else:
            logger.error(f"Unknown spraying strategy type specified: '{strategy_type}'")
            return None

    except KeyError as e:
        logger.error(f"Missing configuration key expected by strategy '{strategy_type}': {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid configuration value for strategy '{strategy_type}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating strategy instance of type '{strategy_type}': {e}")
        logger.error(traceback.format_exc())
        return None