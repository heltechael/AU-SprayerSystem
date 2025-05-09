import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os


import time

from arena_api.callback import callback, callback_function
from arena_api.system import system
from sensor_msgs.msg import Range

import glob
from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix
from std_msgs.msg import String

import shapely
import shapely.geometry
import fiona

from datetime import datetime

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup




class spraying_state_control(Node):

    def __init__(self):
        super().__init__('spraying_state_control')
        
        
        
        timer_period = 0.2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.gps_callback_group = MutuallyExclusiveCallbackGroup()
        
        self.gps_subscription = self.create_subscription(
                    GPSFix,
                    'gpsfix',
                    self.gps_callback,
                    10,
                    callback_group = self.gps_callback_group)
                    
        self.spraying_state_publisher = self.create_publisher(String, 'sprayingState', 10)
                    
        self.current_latitude = 0.
        self.current_longitude = 0.
        
        shapefile_dir = '/media/OREI-cam/shapefiles/'
        
        filenameList = glob.glob(shapefile_dir + '*' + '.shp')
        self.polygon_list = []
        
        for shapefile_path in filenameList:
            with fiona.open(shapefile_path) as shapefile:
                print(shapefile_path)
                # Iterate over the records
                for record in shapefile:
                    # Print the record
                    #print(record)
                    self.polygon_list.append(record)
        print(self.polygon_list)
            

    def timer_callback(self):
        self.check_spraying_state_and_publish()
        
        
    def gps_callback(self, msg):
        self.current_latitude = msg.latitude
        self.current_longitude = msg.longitude
        
        
    def check_spraying_state_and_publish(self):
        currentPosition = shapely.geometry.Point(self.current_longitude, self.current_latitude)
        #im_density = 5.0
        spraying_state = ""
        for individual_polygon in self.polygon_list:
            #print(individual_polygon['geometry'])
            shapely_polygon = shapely.geometry.shape(individual_polygon['geometry'])
            if(shapely_polygon.contains(currentPosition)):
                #force_recording = True
                spraying_state = individual_polygon['properties']['Treatment']
                break
                
        
        spraying_state_msg = String()
        spraying_state_msg.data = spraying_state
        self.spraying_state_publisher.publish(spraying_state_msg)
        #print("Current treatment is: ", spraying_state, " at :", str(self.current_longitude), ", ", str(self.current_latitude))
        

    

def main(args=None):
    rclpy.init(args=args)

    spraying_state_control_node = spraying_state_control()
    

    rclpy.spin(spraying_state_control_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    orei_cam_gui_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
