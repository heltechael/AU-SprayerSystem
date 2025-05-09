import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from rosidl_runtime_py import message_to_ordereddict


import os


import time

from arena_api.callback import callback, callback_function
from arena_api.system import system

from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix
from std_msgs.msg import Empty
from geopy import distance



class camera_trigger(Node):

    def __init__(self):
        super().__init__('camera_trigger')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.i = 0
        self.previous_trigger_position = None
        self.time_at_start = datetime.now()
        self.time_at_last_trigger = datetime.now()
        # Create a device

        self.gps_subscription = self.create_subscription(
                    GPSFix,
                    'gpsfix',
                    self.gps_callback,
                    10)

        self.trigger_publisher = self.create_publisher(Empty, 'trigger', 10)
    
    def date_diff_in_Seconds(self, dt2, dt1):
        # Calculate the time difference between dt2 and dt1
        timedelta = dt2 - dt1
        # Return the total time difference in seconds
        return timedelta.total_seconds() #timedelta.days * 24 * 3600 + timedelta.seconds    

    def gps_callback(self, msg):
        #print("received gps msg..")
        if((msg.latitude > 90) or (msg.latitude < -90) or (msg.longitude > 90) or (msg.longitude < -90)):
            print("no gps reception..")
            #print(str(self.date_diff_in_Seconds(datetime.now(), self.time_at_start)))
            if ( self.date_diff_in_Seconds(datetime.now(), self.time_at_start) < 60):
                if ( self.date_diff_in_Seconds(datetime.now(), self.time_at_last_trigger) > 2):
                    self.time_at_last_trigger = datetime.now()
                    self.trigger_publisher.publish(Empty())
            
            return -1
        if(self.previous_trigger_position == None):
            self.previous_trigger_position = msg
            self.trigger_publisher.publish(Empty())
        else:
            if ( self.date_diff_in_Seconds(datetime.now(), self.time_at_last_trigger) > 0.20):  #was 0.15 for orei sampling   was 0.125 for first farmdroid sampling
                distance_since_last_trigger = distance.great_circle((self.previous_trigger_position.latitude, self.previous_trigger_position.longitude), (msg.latitude, msg.longitude)).m
                if(distance_since_last_trigger > 0.10):  #was 0.35 for orei sampling   was 0.05 for first farmdroid sampling
                    print("Triggering camera..")
                    self.previous_trigger_position = msg
                    self.time_at_last_trigger = datetime.now()
                    self.trigger_publisher.publish(Empty())
                print(distance_since_last_trigger)
            

def main(args=None):
    rclpy.init(args=args)

    camera_trigger_instance = camera_trigger()
    
    #camera_trigger.run()
    rclpy.spin(camera_trigger_instance)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_trigger_instance.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
