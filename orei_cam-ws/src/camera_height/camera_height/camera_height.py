
from numpy_ringbuffer import RingBuffer

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from std_msgs.msg import UInt8MultiArray 
from sensor_msgs.msg import Range
from scipy import stats



class camera_height(Node):

    def __init__(self):
        super().__init__('camera_height')
        
        self.lidar_ringbuffer = RingBuffer(capacity=128, dtype=np.float64)

        
        self.image_number = 0

        self.lidar_subscription = self.create_subscription(
                    Range,
                    'lidar_distance',
                    self.lidar_callback,
                    10)
                    
        
        self.camera_height_publisher = self.create_publisher(Range, 'camera_height', 10)
        
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.estimate_and_publish_camera_height)
        
        
        
    def lidar_callback(self, msg):
        if(msg.range < 10): # sometimes when no return signal is detected, the lidar reports 230 meter distance. This is to filter out those responses..
            self.lidar_ringbuffer.append(msg.range)
            
    
    def estimate_and_publish_camera_height(self):
        camera_height_estimate = stats.trim_mean(np.array(self.lidar_ringbuffer), 0.2) - 0.07
        new_range_msg = Range()
        new_range_msg.range = camera_height_estimate
        self.camera_height_publisher.publish(new_range_msg)
    
    
def main(args=None):
    rclpy.init(args=args)

    lidar_analyser_node = camera_height()
    
    #camera_node.run()
    #example_entry_point()
    rclpy.spin(lidar_analyser_node)
    
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_analyser_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

