import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os


import time

from arena_api.callback import callback, callback_function
from arena_api.system import system
from sensor_msgs.msg import Range

from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix
from sensor_msgs.msg import Range

from sensor_msgs.msg import Image # Image is the message type

from orei_cam_interfaces.msg import ImageWithMetadata
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library



class orei_cam_gui(Node):

    def __init__(self):
        super().__init__('orei_cam_gui')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.i = 0
        self.bridge = CvBridge()
        self.have_received_first_image = False
        self.image_data = ImageWithMetadata()
        #self.image_content = None
        
        self.current_camera_height = Range()
        
        self.gps_reception = False
        
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        
        
        
        self.image_subscriber = self.create_subscription(
            ImageWithMetadata, 
            'images', 
            self.image_received_callback, 
            10)
            
        self.camera_height_subscriber = self.create_subscription(
            Range, 
            'camera_height', 
            self.camera_height_received_callback, 
            10)

    def timer_callback(self):
        #if self.have_received_first_image:
        self.displayImageInColor()
        

    def image_received_callback(self, msg):
        if((msg.latitude > 90) or (msg.latitude < -90) or (msg.longitude > 90) or (msg.longitude < -90)):
            self.gps_reception = False
        else:
            self.gps_reception = True
        if((msg.image_number % 4)==0):
            self.image_data = msg
        #self.image_content = self.bridge.imgmsg_to_cv2(msg.image)
        self.have_received_first_image = True
        
    def camera_height_received_callback(self, msg):
        self.current_camera_height = msg
        
    def getAvailableStorageInGB(self):
        target_dir = "/media/OREI-cam/images/"
        statvfs = os.statvfs(target_dir)
        return statvfs.f_frsize * statvfs.f_bavail / 1024 / 1024 / 1024
        
        
    #def adjust_gamma(self, image, gamma=1.5):
    #  	# build a lookup table mapping the pixel values [0, 255] to
    #  	# their adjusted gamma values
    #  	invGamma = 1.0 / gamma
    #  	table = np.array([((i / 255.0) ** invGamma) * 255
    #  		for i in np.arange(0, 256)]).astype("uint8")
    #  	# apply gamma correction using the lookup table
    #  	return cv2.LUT(image, table)

    def displayImageInColor(self):
        #print("Displaying image..")
        #global lat, lon, alt
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        #colour_image = None
        colour_image_gamma_adjusted = np.zeros((1712, 1980,3))
        if self.have_received_first_image:
            colour_image_gamma_adjusted = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.image_data.image), cv2.COLOR_BAYER_RG2RGB)
            colour_image_gamma_adjusted = cv2.resize(colour_image_gamma_adjusted, (1980, 1712))
            #colour_image_bordered = cv2.copyMakeBorder(colour_image, 0, 0, 165*2, 0,cv2.BORDER_CONSTANT,value=[0,0,0])
            #colour_image_gamma_adjusted = colour_image  #self.adjust_gamma(colour_image_bordered, gamma=2.0)
            
            #del colour_image
        #else:
        #    colour_image_gamma_adjusted = np.zeros((1128, 1980,3))
        
        #lidar_plot = getLidarPlot()
        #plot_size = lidar_plot.shape
        #colour_image_gamma_adjusted[:,0:plot_size[1],:] = lidar_plot
        # Add textual info on the status screen
        mid_string = ""
        
        if (self.gps_reception is False):
            mid_string = mid_string + "NO GPS RECEPTION"
        top_string = f"Im: {self.image_data.image_number:.0f}"
        #if(recording_active):
        #    top_string = top_string + " REC! "
        #else:
        #    top_string = top_string + " NOT REC! "
        #mid_string = ""
        #if(backup_in_progress):
        #    mid_string = mid_string + "TRANSFERRING! "
        #    path, dirs, files = next(os.walk("/media/storage/images"))
        #    files_to_transfer = len(files)
        #    path, dirs, files = next(os.walk("/media/nvidia/HVCam/images"))
        #    current_files = len(files) - files_before_transfer_dest
        #    mid_string = mid_string + str(current_files) + "/" + str(files_before_transfer)
        #if(sync_in_progress):
        #    mid_string = mid_string + "Syncing - Don't unplug!"
        top_string = top_string + ",  " + str(np.round(self.getAvailableStorageInGB()).astype(int)) + " GB"
        #if (getAvailableStorageInGB()<80):
        #    mid_string = mid_string + " LOW STORAGE! "
        cv2.putText(colour_image_gamma_adjusted, top_string, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0,255), 18)
        cv2.putText(colour_image_gamma_adjusted, top_string, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255,255), 12)
        cv2.putText(colour_image_gamma_adjusted, mid_string, (80, (700-100)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0,255), 18)
        cv2.putText(colour_image_gamma_adjusted, mid_string, (80, (700-100)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255,255), 12)
        if (self.image_data.speed*3.6 < 100): # Check for data validity
            current_speed = self.image_data.speed*3.6
        else:
            current_speed = 99.9
        bot_string = f"{current_speed:.1f} km/t,  {self.current_camera_height.range*100:.1f}/100 cm"
        
        cv2.putText(colour_image_gamma_adjusted, bot_string, (100, 1150-100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0,255), 18)
        cv2.putText(colour_image_gamma_adjusted, bot_string, (100, 1150-100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255,255), 12)
        
        #cv2.imwrite(str(self.image_data.image_number) + ".png", colour_image_gamma_adjusted) 
        cv2.imshow("window", colour_image_gamma_adjusted)
        cv2.waitKey(100)
        del colour_image_gamma_adjusted
    

def main(args=None):
    rclpy.init(args=args)

    orei_cam_gui_node = orei_cam_gui()
    

    rclpy.spin(orei_cam_gui_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    orei_cam_gui_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
