import rclpy
from rclpy.node import Node
from rclpy.time import Time
import struct

from std_msgs.msg import String
from std_msgs.msg import UInt8MultiArray 
from sensor_msgs.msg import Range


class lidar_driver(Node):

    def __init__(self):
        super().__init__('lidar_driver')

        
        self.image_number = 0

        self.serial_subscription = self.create_subscription(
                    UInt8MultiArray,
                    '/serial_read',
                    self.serial_callback,
                    10)
                    
        
        self.lidar_publisher = self.create_publisher(Range, 'lidar_distance', 10)
        
        
        
    def serial_callback(self, msg):
        
        vstr = ''.join([chr(k) for k in msg.data])
        #print(vstr)
        print("Got new msg..")
        if(len(vstr)==8):
            new_range_msg = Range()
            float_range = float(vstr[:-4])
            new_range_msg.range = float_range
            self.lidar_publisher.publish(new_range_msg)
        #print(vstr)
        #print(vstr[:-1])
        #print(vstr[:-2])
        #print(vstr[:-3])
        #print(vstr[:-4])
         #struct.unpack("f", bytes(vstr[:-2]))
       # print(float_range)
        
        
def main(args=None):
    rclpy.init(args=args)

    serial_node = lidar_driver()
    
    #camera_node.run()
    #example_entry_point()
    rclpy.spin(serial_node)
    
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    serial_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
