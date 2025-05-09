import serial
import config


class NozzleActivation:
    def __init__(self):
        # Serial communication parameters from config.py
        self.PORT_NAME = config.PORT_NAME
        self.BAUD_RATE = config.BAUD_RATE
        self.TIMEOUT = config.TIMEOUT

    def generate_hex_number(self, binary_number):
        """
        Generates a serial message based on a binary number.

        :param binary_number: A binary string representing relay states (e.g., "1010101010101010101010101010101").
        :return: A hex activation sequence without the "0x" prefix and filled with zeros to 8 characters, if the binary number hasn't 32 characters returns "00000000".
        """
        try:
            if len(binary_number) != 32:
                raise ValueError("The binary number MUST have 32 characters.")
            else:
                hex_activation_sequence = hex(int(binary_number, 2))
                return hex_activation_sequence[2:].zfill(8)
        except ValueError as e:
            print(f"Error: {e}")
            return "00000000"  # FIXME is it ok? all off in case of error

    def send_command(self, binary_number):
        """
        Sends a command via the serial port.

        :param binary_number: A binary string representing ALL relay states (e.g., "1010101010101010101010101010101").
        """
        try:
            ser = serial.Serial(
                self.PORT_NAME, self.BAUD_RATE, timeout=self.TIMEOUT)
            if ser.is_open:
                print(f"Connected to {self.PORT_NAME}")
                message = "relay writeall " + \
                    self.generate_hex_number(binary_number) + "\n\r"
                ser.write(message.encode())  # Convert the string to bytes
                print(f"Command sent: {message}")
                ser.close()
        except serial.SerialException as e:
            print(f"Serial communication error: {e}, " +
                  "MAYBE the port is already in use by another application or the port name is incorrect/does not exist.")


# Test the class
if __name__ == "__main__":
    nozzle = NozzleActivation()
    binary_test = "01110000000000000000000000000000"
    nozzle.send_command(binary_test)
