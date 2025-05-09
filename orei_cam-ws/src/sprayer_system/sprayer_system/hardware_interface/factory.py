from typing import Dict, Any, Optional
from .base_interface import BaseHardwareInterface
from .serial_relay_driver import SerialRelayInterface

def create_hardware_interface(interface_type: str, config: Dict[str, Any], logger) -> Optional[BaseHardwareInterface]:
    logger.info(f"[HardwareFactory] Attempting to create interface: '{interface_type}'")
    interface_type_lower = interface_type.lower() 

    if interface_type_lower == "serial_relay":
        serial_config = config.get('serial_relay', {})
        if not serial_config:
             logger.error("Hardware type is 'serial_relay' but no 'serial_relay' config section found.")
             return None
        try:
            return SerialRelayInterface(serial_config, logger)
        except Exception as e:
            logger.fatal(f"Failed to initialize SerialRelayInterface: {e}")
            return None
    elif interface_type_lower == "dummy":
        logger.info("Creating Dummy Hardware Interface (Not Implemented).")
        return None
    else:
        logger.error(f"Unknown hardware interface type: '{interface_type}'")
        return None