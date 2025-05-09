import logging
import time
from typing import List, Optional, Dict, Any
import serial 

from .base_interface import BaseHardwareInterface

class SerialRelayInterface(BaseHardwareInterface):
    """
    Hardware interface for controlling relays via serial commands.

    Formats a boolean nozzle state list (up to 25 nozzles) and a forward
    velocity (cm/s, 0-127) into the required 32-bit integer representation:
    - Bits 31-25 (MSB 7 bits): Velocity in cm/s (0-127)
    - Bits 24-0 (LSB 25 bits): Relay states (Bit 0 = Nozzle 0)
    Sends this combined state over the serial port
    """
    COMMAND_PREFIX = ""
    COMMAND_TERMINATOR = "\n"
    EXPECTED_TOTAL_BITS = 32
    EXPECTED_HEX_PAYLOAD_LENGTH = EXPECTED_TOTAL_BITS // 4 # 8 hex chars

    NUM_VELOCITY_BITS = 7
    NUM_RELAY_BITS = 25 
    VELOCITY_SHIFT = NUM_RELAY_BITS # Shift velocity left by 25 bits

    MAX_VELOCITY_PAYLOAD = (1 << NUM_VELOCITY_BITS) - 1 # 127
    MAX_RELAY_STATE_INT = (1 << NUM_RELAY_BITS) - 1
    MAX_SUPPORTED_NOZZLES = NUM_RELAY_BITS # Can control up to 25 nozzles

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self._logger = logger
        self._port: Optional[str] = None
        self._baudrate: int = 9600
        self._num_hardware_bits: int = self.EXPECTED_TOTAL_BITS
        self._serial_connection: Optional[serial.Serial] = None
        self._last_sent_combined_state: Optional[int] = None # Cache last *combined* state

        try:
            self._port = config.get('port')
            if not self._port:
                raise ValueError("'port' not specified in serial_relay config.")
            self._baudrate = int(config.get('baudrate', 9600))

            self._logger.info(f"SerialRelayInterface configured for port {self._port} at {self._baudrate} baud.")
            self._logger.info(f"  Hardware Command Format: {self.EXPECTED_TOTAL_BITS}-bit total")
            self._logger.info(f"    Bits 31-{self.NUM_RELAY_BITS}: {self.NUM_VELOCITY_BITS}-bit Velocity (0-{self.MAX_VELOCITY_PAYLOAD} cm/s)")
            self._logger.info(f"    Bits {self.NUM_RELAY_BITS-1}-0: {self.NUM_RELAY_BITS}-bit Relay States (Max {self.MAX_SUPPORTED_NOZZLES} relays)")

        except (ValueError, TypeError, KeyError) as e:
            self._logger.fatal(f"Invalid serial_relay configuration: {e}")
            raise ValueError(f"SerialRelayInterface config error: {e}") from e

    def connect(self) -> bool:
        if self.is_connected():
            self._logger.info("Serial port already connected.")
            return True
        if not self._port:
            self._logger.error("Cannot connect: Serial port not configured.")
            return False
        try:
            self._logger.info(f"Attempting serial connection to {self._port}...")
            self._serial_connection = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=0.1, 
                write_timeout=0.1
            )
            time.sleep(0.1) # let port settle
            if self._serial_connection.is_open:
                self._serial_connection.reset_input_buffer()
                self._serial_connection.reset_output_buffer()
                self._logger.info(f"Successfully connected to serial port {self._port}.")
                self.send_combined_integer_state(0) # all off on connect
                return True
            else:
                self._logger.error(f"Failed to open serial port {self._port} (is_open=False).")
                self._serial_connection = None
                return False
        except serial.SerialException as e:
            self._logger.error(f"Serial error connecting to {self._port}: {e}")
            self._serial_connection = None
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error connecting to {self._port}: {e}")
            self._serial_connection = None
            return False

    def disconnect(self):
        if self._serial_connection and self._serial_connection.is_open:
            try:
                self.send_combined_integer_state(0) # all off on disconnect
                time.sleep(0.05) 
                self._logger.info(f"Disconnecting from serial port {self._port}...")
                self._serial_connection.close()
                self._logger.info("Serial port disconnected.")
            except Exception as e:
                self._logger.error(f"Error during serial disconnection: {e}")
        self._serial_connection = None
        self._last_sent_combined_state = None # clear cache on disconnect

    def is_connected(self) -> bool:
        return self._serial_connection is not None and self._serial_connection.is_open

    def set_nozzle_state(self, state: List[bool], velocity_cmps: int):
        if not self.is_connected():
            self._logger.error("Cannot set nozzle state: Not connected.")
            return

        num_commanded_nozzles = len(state)

        clamped_velocity = max(0, min(int(velocity_cmps), self.MAX_VELOCITY_PAYLOAD))
        if clamped_velocity != velocity_cmps:
             self._logger.warning(f"Input velocity {velocity_cmps} cm/s clamped to {clamped_velocity} for 7-bit payload.", throttle_duration_sec=5)

        relay_state_int = 0
        if num_commanded_nozzles > self.MAX_SUPPORTED_NOZZLES:
             self._logger.warning(f"Commanded nozzle list length ({num_commanded_nozzles}) exceeds hardware limit ({self.MAX_SUPPORTED_NOZZLES}). Truncating.")
             state = state[:self.MAX_SUPPORTED_NOZZLES] # Truncate the list

        for i, is_on in enumerate(state):
            if is_on:
                relay_state_int |= (1 << i) # Set the i-th bit

        combined_state_int = (clamped_velocity << self.VELOCITY_SHIFT) | relay_state_int

        if combined_state_int == self._last_sent_combined_state:
            # self._logger.debug("Combined state unchanged, skipping serial send.")
            return

        if self.send_combined_integer_state(combined_state_int):
             self._last_sent_combined_state = combined_state_int 
        else:
             self._logger.error("Failed to send combined nozzle state command.")
             self._last_sent_combined_state = None 

    def _format_command(self, hex_payload: str) -> Optional[bytes]:
        if len(hex_payload) != self.EXPECTED_HEX_PAYLOAD_LENGTH:
            self._logger.error(f"Internal Error: Formatted hex payload length ({len(hex_payload)}) != expected ({self.EXPECTED_HEX_PAYLOAD_LENGTH}). Payload: '{hex_payload}'")
            return None
        if not all(c in '0123456789abcdefABCDEF' for c in hex_payload):
             self._logger.error(f"Internal Error: Invalid characters in hex payload '{hex_payload}'.")
             return None

        command_str = f"{self.COMMAND_PREFIX}{hex_payload}{self.COMMAND_TERMINATOR}"
        return command_str.encode('ascii')

    def _send_raw_command(self, command_bytes: bytes) -> bool:
        if not self.is_connected():
            self._logger.error("Cannot send raw command: Not connected.")
            return False
        if not command_bytes:
            self._logger.error("Cannot send empty raw command.")
            return False

        try:
            bytes_written = self._serial_connection.write(command_bytes)
            # self._logger.debug(f"Serial command sent: {command_bytes!r} ({bytes_written} bytes)")
            return True
        except serial.SerialTimeoutException:
            self._logger.error("Serial write operation timed out.")
            return False
        except serial.SerialException as e:
            self._logger.error(f"Serial communication error during send: {e}")
            self.disconnect() # disconnect on error to force reconnect attempt later
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error during serial send: {e}")
            return False

    def send_combined_integer_state(self, combined_state_int: int) -> bool:
        max_val = (1 << self._num_hardware_bits) - 1
        if not (0 <= combined_state_int <= max_val):
            # This check should ideally not fail if inputs to set_nozzle_state are correct
            self._logger.error(f"Internal Error: Combined integer state {combined_state_int} out of range for {self._num_hardware_bits} bits (0-{max_val}).")
            return False
        try:
            # format as hex, 0-padded to the expected length (8 chars for 32 bits)
            hex_payload = f'{combined_state_int:0{self.EXPECTED_HEX_PAYLOAD_LENGTH}x}'
            command_bytes = self._format_command(hex_payload)
            if command_bytes is None:
                return False
            return self._send_raw_command(command_bytes)
        except Exception as e:
            self._logger.error(f"Error formatting combined integer state {combined_state_int}: {e}")
            return False