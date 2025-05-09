import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String

import os


import time
import glob
import datetime
import shutil
import ntpath

import subprocess

from arena_api.callback import callback, callback_function
from arena_api.system import system

from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix

from sensor_msgs.msg import Image # Image is the message type

import cv2 # OpenCV library
from concurrent.futures import ThreadPoolExecutor




#rsync -rvz -e 'ssh -p 2022' --progress manual_trigger__2024-03-28\ 16-01-51.800__0__54.9514874383__10.2296287867__0.7.png hvcam@skovsenhub.myds.me:/volume3/HVCam/HVCam3/



executor = ThreadPoolExecutor(max_workers=6)

class image_uploader(Node):

    def __init__(self):
        super().__init__('image_uploader')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.i = 0
        
        self.source_image_dir = '/media/OREI-cam/images/'
        self.offload_local_dir = '/media/OREI-cam/uploaded_images/'
        self.image_type = '.png'
        self.dest_port = '2022'  # hardcoded below
        self.dest_username = 'hvcam'
        self.dest_hostname = 'skovsenhub.myds.me'
        self.dest_path = ':/volume3/HVCam/orei_cam/'
        
        
    def date_diff_in_Seconds(self, dt2, dt1):
        # Calculate the time difference between dt2 and dt1
        timedelta = dt2 - dt1
        # Return the total time difference in seconds
        return timedelta.days * 24 * 3600 + timedelta.seconds


    def run(self):
        while(True):
            time.sleep(1)
            filenameList = glob.glob(self.source_image_dir + '**/' + '*' + self.image_type)
            filenameList.sort(reverse = True)
            #if len(filenameList) > 0:
            for filename in filenameList:
                gps_file_stamp = filename.split('__')[0].split('/')[-1]
                #print(gps_file_stamp)
                datetime_of_image = datetime.fromtimestamp(float(gps_file_stamp) + 315964782 )
                current_datetime = datetime.now()
                #print(datetime_of_image)
                #print(current_datetime)
                #print(filename.split('/')[-2])
                
                #print(filename)
                #print(filedate)
                #date1 = datetime.strptime(filedate, '%Y-%m-%d %H-%M-%S.%f')
                #date2 = datetime.now()
                
                if(self.date_diff_in_Seconds(current_datetime, datetime_of_image)>20):
                    # Run rsync command to upload selected image
                    #args_csv = ["rsync", "-rvz", "-essh -p 2022", filename]
                    #args_csv.append(self.dest_username + "@" + self.dest_hostname + self.dest_path + filename.split('/')[-2] + "/")
                    
                    #args_image = ["rsync", "-rvz", "-essh -p 2022", filename]
                    #args_image.append(self.dest_username + "@" + self.dest_hostname + self.dest_path + filename.split('/')[-2] + "/")
                    
                    args_all = ["rsync", "-rvz", "-essh -p 2022", filename[:-3]+'csv', filename]
                    args_all.append(self.dest_username + "@" + self.dest_hostname + self.dest_path + filename.split('/')[-2] + "/")
                    
                    
                    
                    print("Uploading csv and image file..")
                    #print(args_all)
                    try:
                        subprocess_status = subprocess.call(args_all, timeout=60)
                      #except subprocess.TimeoutExpired:
                    except:
                        print("Upload had timeout..")
                    print(subprocess_status)
                    if(subprocess_status == 0):
                        offload_dir = self.offload_local_dir + str(datetime_of_image.date()) + "/"
                        os.makedirs(offload_dir, exist_ok=True)
                        filename_without_path = ntpath.basename(filename)
                        shutil.move(filename[:-3]+'csv', offload_dir + filename_without_path[:-3]+'csv')
                        shutil.move(filename, offload_dir + filename_without_path)
                #    # only upload the currently newest file. Then check for new files..
                    break
            
            if(len(filenameList)==0):
                time.sleep(10)

def main(args=None):
    rclpy.init(args=args)

    image_uploader_node = image_uploader()
    image_uploader_node.run()
    #camera_node.run()
    #example_entry_point()
    rclpy.spin(image_uploader_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_uploader_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
