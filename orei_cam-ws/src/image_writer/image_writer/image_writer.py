import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from rosidl_runtime_py import message_to_ordereddict
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import os
import time
import rclpy
import logging
from arena_api.callback import callback, callback_function
from arena_api.system import system
from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image # Image is the message type
from orei_cam_interfaces.msg import ImageWithMetadata
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from concurrent.futures import ThreadPoolExecutor
import random
import string

executor = ThreadPoolExecutor(max_workers=12)

class image_writer(Node):

    def __init__(self):
        super().__init__('image_writer')
        self.bridge = CvBridge()
        self.image_writer_callback_group = MutuallyExclusiveCallbackGroup() #ReentrantCallbackGroup()
        self.random_session_string = ''.join(random.choices(string.ascii_letters, k=5))

        self.image_subscriber = self.create_subscription(
            ImageWithMetadata,
            'images',
            self.image_received_callback,
            20,
            callback_group = self.image_writer_callback_group)
        
        self.get_logger().info('ImageWriter node initialized.')

    def image_received_callback(self, msg):
        print("subscriber has received an image!")
        image_data = self.bridge.imgmsg_to_cv2(msg.image).copy()
        gps_date = time.strftime('%Y-%m-%d', time.gmtime(msg.gpsfix.time + 315964800 )) # Add rough offset between gpstime and unix epochs
        dir_path = "/media/OREI-cam/images/" + gps_date + "/"
        os.makedirs(dir_path, exist_ok=True)

        img_path = dir_path + "{:.2f}".format(msg.gpsfix.time) + "__" + "{:06d}".format(msg.image_number) + "__lat_" + str(msg.latitude) + "__lon_" + str(msg.longitude) + "__track_" + "{:.2f}".format(msg.track) + "__" + self.random_session_string + ".png"
        a = executor.submit(cv2.imwrite,img_path,image_data)
        a = executor.submit(self.save_msg_data,dir_path, msg)

        self.get_logger().debug(f'Saved image and metadata: {img_path}')

    def save_msg_data(self, dir_path, msg):
        msg_dict = message_to_ordereddict(msg, truncate_length=100)
        metadata_path = dir_path + "{:.2f}".format(msg.gpsfix.time) + "__" + "{:06d}".format(msg.image_number) + "__lat_" + str(msg.latitude) + "__lon_" + str(msg.longitude) + "__track_" + "{:.2f}".format(msg.track) + "__" + self.random_session_string + ".csv"
        with open(metadata_path, 'w') as f:
            for key,val in msg_dict.items():
                f.write('{},{}\n'.format(key, val))
        self.get_logger().debug(f'Saved metadata: {metadata_path}')

def main(args=None):
    rclpy.init(args=args)

    image_writer_instance = image_writer()
    executor = MultiThreadedExecutor()
    executor.add_node(image_writer_instance)

    try:
        image_writer_instance.get_logger().info('image_writer node is running.')
        executor.spin()
    except KeyboardInterrupt:
        image_writer_instance.get_logger().info('Keyboard interrupt, shutting down image_writer.\n')

    image_writer_instance.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
