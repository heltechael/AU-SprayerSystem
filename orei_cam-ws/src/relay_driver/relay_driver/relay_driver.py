import rclpy
from rclpy.node import Node
from rclpy.time import Time
import struct
import time
import random

from std_msgs.msg import String
from std_msgs.msg import UInt8MultiArray 
from sensor_msgs.msg import Range


class relay_driver(Node):

    def __init__(self):
        super().__init__('relay_driver')


        #self.serial_subscription = self.create_subscription(
        #            UInt8MultiArray,
        #            '/serial_read',
        #            self.serial_callback,
        #            10)
                    
        
        self.relay_publisher = self.create_publisher(UInt8MultiArray, '/relay_serial_driver/serial_write', 10)
        
    def set_all_relays(self, relay_pattern):
        print("setting relays..")
        serial_msg = UInt8MultiArray()
        hex_string = "{0:0{1}x}".format(relay_pattern,8)
        print(hex_string)
        string_to_send = hex_string+'\n'
        serial_msg.data = str.encode(string_to_send)
        print(serial_msg.data)
        self.relay_publisher.publish(serial_msg)
        
    #def serial_callback(self, msg):
    #    
    #    vstr = ''.join([chr(k) for k in msg.data])
    #    #print(vstr)
    #    print("Got new msg..")
    #    if(len(vstr)==8):
    #        new_range_msg = Range()
    #        float_range = float(vstr[:-4])
    #        new_range_msg.range = float_range
    #        self.lidar_publisher.publish(new_range_msg)
        #print(vstr)
        #print(vstr[:-1])
        #print(vstr[:-2])
        #print(vstr[:-3])
        #print(vstr[:-4])
         #struct.unpack("f", bytes(vstr[:-2]))
       # print(float_range)
        
        
def main(args=None):
    rclpy.init(args=args)

    relay_driver_node = relay_driver()
    print("something is running..")
    #camera_node.run()
    #example_entry_point()
    while(1):
    
        relay_driver_node.set_all_relays(65535)
        #time.sleep(5)
        relay_driver_node.set_all_relays(0)
        #time.sleep(1)
        #relay_driver_node.set_all_relays()
        rclpy.spin(relay_driver_node)
    
    
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    relay_driver_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
