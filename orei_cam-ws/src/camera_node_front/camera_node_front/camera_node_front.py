import rclpy
from rclpy.node import Node
from rclpy.time import Time
import ctypes

#libc = ctypes.CDLL('libc.so.6')


from std_msgs.msg import String
from std_msgs.msg import Empty
from orei_cam_interfaces.msg import ImageWithMetadata
from sensor_msgs.msg import Range

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup



import time

from arena_api.callback import callback, callback_function
from arena_api.system import system

from arena_api.buffer import BufferFactory
from arena_api import enums
import numpy as np
from datetime import datetime
from gps_msgs.msg import GPSFix

from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library




def create_devices_with_tries():
    '''
    This function waits for the user to connect a device before raising
        an exception
    '''

    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{sec_count + 1 } seconds passed ',
                    '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'Created {len(devices)} device(s)')
            return devices
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')


# Must have the decorator on the callback function
@callback_function.node.on_update
def print_node_value_exposure_start(node, *args, **kwargs):

    print(f'Message from callback')
    print(f'\'{node.name}\' event has triggered this callback')
    print(datetime.now())
    print(rclpy.clock.Clock().now())
    camera_node.ros_time_for_most_recent_trigger = camera_node.get_clock().now()
    #image_buffer = None
    #try:
    #    image_buffer = self.device.get_buffer(timeout=500)
    #    print(f' Width X Height = '
    #            f'{image_buffer.width} x {image_buffer.height}')
        
    #except:
    #    print("Could not read image from buffer..")
    #if image_buffer is not None:

        #item = BufferFactory.copy(image_buffer)
        
        #data = np.ctypeslib.as_array(item.pdata, shape=(len(item.data),)).astype(np.uint16)
        #if not "result" in locals():
        #    result = np.empty(
        #        data.size * 2 // 3, np.uint16
        #    )  # Initialize matrix for storing the pixels. We use 1.5 bytes per pixel

            # 12 bits packing: ######## ######## ########
            #                  | 8bits| | 4 | 4  |  8   |
            #                  |  lsb0 | |lsb1|msb0 |  msb1 |
            #                  <-----------><----------->
            #                     12 bits       12 bits

            # MSB0                  #LSB0
        #result[0::2] = ((data[1::3] & 15) << 8) | data[0::3]
        # LSB1              #MSB1
        #result[1::2] = (data[1::3] >> 4) | (data[2::3] << 4)

        #npndarray = np.reshape(result, (item.height, item.width))
        #encoding = "16UC1"
        #print(npndarray.shape)
    #    self.device.requeue_buffer(image_buffer)
        #BufferFactory.destroy(item)
		
    #image_buffer = self.device.get_buffer()
    #print(f' Width X Height = '
	#		f'{image_buffer.width} x {image_buffer.height}')
    
def apply_trigger_settings(device):

    '''
    Enable trigger mode before setting the source and selector
        and before starting the stream. Trigger mode cannot be turned on and off
        while the device is streaming.
    '''

    # Make sure Trigger Mode set to 'Off' after finishing this example
    device.nodemap.get_node('TriggerMode').value = 'On'

    '''
    Set the trigger source to software in order to trigger buffers
        without the use of any additional hardware. Lines of the GPIO can also be
        used to trigger.
    '''
    device.nodemap.get_node('TriggerSource').value = 'Software'

    device.nodemap.get_node('TriggerSelector').value = 'FrameStart'

    device.tl_stream_nodemap.get_node(
        'StreamBufferHandlingMode').value = 'NewestOnly'


class camera_front_node(Node):

    def __init__(self):
        super().__init__('camera_front_node')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.i = 0
        
        # Create a device
        devices = create_devices_with_tries()
        self.device = devices[0]
        self.active_acq_waiting_for_gps = False
        self.ros_time_for_most_recent_trigger = self.get_clock().now()#rclpy.clock.Clock().now()
        self.previous_gps_msg_for_acq = GPSFix()
        self.current_image_msg_to_publish = ImageWithMetadata()
        self.previous_latency_from_gps_to_exposure = rclpy.duration.Duration()
        
        self.gps_callback_group = MutuallyExclusiveCallbackGroup()
        self.trigger_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.camera_height_callback_group = MutuallyExclusiveCallbackGroup()
        self.image_number = 0
        self.current_camera_height = Range()
    
        # Store nodes' initial values ---------------------------------------------

        # get node values that will be changed in order to return their values at
        # the end of the example
        streamBufferHandlingMode_initial = \
            self.device.tl_stream_nodemap['StreamBufferHandlingMode'].value
        triggerSource_initial = self.device.nodemap['TriggerSource'].value
        triggerSelector_initial = self.device.nodemap['TriggerSelector'].value
        triggerMode_initial = self.device.nodemap['TriggerMode'].value
        

        # -------------------------------------------------------------------------

        print(f'Device used in the example:\n\t{self.device}')

        apply_trigger_settings(self.device)
        
        self.device.nodemap.get_node('AcquisitionMode').value = 'Continuous'
        self.device.nodemap.get_node('GevSCPSPacketSize').value = 8966
        
        
        #self.device.nodemap.get_node('UserOutputSelector').value = 'UserOutput0'
        
        
        
        
        '''
        Initialize events
            Turn event notification on Select the event type to be notified about
        '''
        #self.device.initialize_events()
        #self.device.nodemap.get_node('EventSelector').value = 'ExposureEnd'
        #self.device.nodemap.get_node('EventNotification').value = 'Off'
        
        #self.device.nodemap.get_node('EventSelector').value = 'ExposureStart'
        #self.device.nodemap.get_node('EventNotification').value = 'On'

        '''
        Register the callback on the node
            Allows for manual triggering
        '''
        #event_node_exposureEnd = self.device.nodemap.get_node('EventExposureEnd')
        #self.handle_exposureEnd = callback.register(event_node_exposureEnd, print_node_value)
        #print(f'Registered \'{print_node_value.__name__}\' function '
        #    f'on {event_node_exposureEnd.name}\' node')
            
        #event_node_exposureStart = self.device.nodemap.get_node('EventExposureStart')
        #self.handle_exposureStart = callback.register(event_node_exposureStart, print_node_value_exposure_start)
        #print(f'Registered \'{print_node_value_exposure_start.__name__}\' function '
        #    f'on {event_node_exposureStart.name}\' node')

        # Get device stream nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap

        # Enable stream auto negotiate packet size
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = False

        # Enable stream packet resend
        tl_stream_nodemap['StreamPacketResendEnable'].value = True
        
        
        #tl_stream_nodemap['DeviceLinkThroughputLimitMode'].value = 'Off'
        self.device.nodemap.get_node('DeviceLinkThroughputLimitMode').value = 'Off'
        
        self.device.nodemap.get_node('PixelFormat').value = 'BayerRG8'
        self.device.nodemap.get_node('AcquisitionFrameRateEnable').value = False
        #self.device.nodemap.get_node('AcquisitionFrameRate').value = float(17)
        
        
        
        #user_set_selector_node = self.device.nodemap.get_node("UserSetSelector")
        #user_set_selector_node.value = "UserSet1"
        #self.device.nodemap.get_node("UserSetLoad").execute()
        
        self.device.initialize_events()
    
        self.device.nodemap.get_node('EventSelector').value = 'ExposureStart'
        self.device.nodemap.get_node('EventNotification').value = 'On'
        #self.device.nodemap.get_node('EventNotification').value = 'Off'
        event_node_exposureStart = self.device.nodemap.get_node('EventExposureStart')
        self.handle_exposureStart = callback.register(event_node_exposureStart, print_node_value_exposure_start)
        print(f'Registered \'{print_node_value_exposure_start.__name__}\' function '
            f'on {event_node_exposureStart.name}\' node')
            
            
        self.gps_subscription = self.create_subscription(
                    GPSFix,
                    'gpsfix',
                    self.gps_callback,
                    10,
                    callback_group = self.gps_callback_group)
                    
                    
        #timer_period = 0.28  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback, callback_group = self.timer_callback_group)
        self.i = 0
        
        
        
        self.trigger_publisher = self.create_publisher(Empty, 'trigger', 10)
        self.image_publisher = self.create_publisher(ImageWithMetadata, 'images', 20)
        
        self.bridge = CvBridge()
        
        self.trigger_subscription = self.create_subscription(
                    Empty,
                    'trigger',
                    self.trigger_callback,
                    10,
                    callback_group = self.trigger_callback_group)
                    
    
    
            
        self.camera_height_subscriber = self.create_subscription(
            Range, 
            'camera_height', 
            self.camera_height_received_callback, 
            10,
            callback_group = self.camera_height_callback_group)
        
    def camera_height_received_callback(self, msg):
        self.current_camera_height = msg
        
    def timer_callback(self):
        print("Triggering camera..")
        self.trigger_publisher.publish(Empty())
        #self.trigger()
        
        
    def trigger_callback(self ,msg):
        print("Received trigger signal..")
        self.trigger()
        self.image_number = self.image_number+1
                    
    def gps_callback(self, msg):
        
        #print("Got gps msg...")
        #print(msg.header.stamp)
        #print(self.active_acq_waiting_for_gps)
        if(self.active_acq_waiting_for_gps):
            print("looking for gps msgs..")
            #if (self.ros_time_for_most_recent_trigger.nanoseconds < Time.from_msg(msg.header.stamp).nanoseconds):
            ### If incoming gps message is newer than the exposure event, use the message just before.
            #print(type(self.ros_time_for_most_recent_trigger))
            #print((self.ros_time_for_most_recent_trigger))
            #print(type(Time.from_msg(msg.header.stamp)))
            #print((Time.from_msg(msg.header.stamp)))
            print("duration..")
            #print(self.ros_time_for_most_recent_trigger - Time.from_msg(msg.header.stamp))
            #print(type(self.ros_time_for_most_recent_trigger - Time.from_msg(msg.header.stamp)))
            #if (self.ros_time_for_most_recent_trigger < Time.from_msg(msg.header.stamp)):
            #    print("Found most recent one..")
                #print(msg.header.stamp)
                # populate image metadata with info
                
            self.previous_gps_msg_for_acq = msg
            self.previous_latency_from_gps_to_exposure = self.ros_time_for_most_recent_trigger - Time.from_msg(msg.header.stamp)
            
            self.current_image_msg_to_publish.latency_from_gps_to_exposure = self.previous_latency_from_gps_to_exposure.to_msg()
            self.current_image_msg_to_publish.gpsfix = self.previous_gps_msg_for_acq
            self.current_image_msg_to_publish.latitude = self.current_image_msg_to_publish.gpsfix.latitude
            self.current_image_msg_to_publish.longitude = self.current_image_msg_to_publish.gpsfix.longitude
            self.current_image_msg_to_publish.altitude = self.current_image_msg_to_publish.gpsfix.altitude
            self.current_image_msg_to_publish.yaw = self.current_image_msg_to_publish.gpsfix.dip
            self.current_image_msg_to_publish.track = self.current_image_msg_to_publish.gpsfix.track
            self.current_image_msg_to_publish.speed = self.current_image_msg_to_publish.gpsfix.speed
            self.current_image_msg_to_publish.time_from_gps_msg_to_im_exposure_in_useconds = float(self.previous_latency_from_gps_to_exposure.nanoseconds/1000)
            
            if(self.previous_latency_from_gps_to_exposure.nanoseconds < 10000000): # if less than 1 1/100 of a second.
                self.active_acq_waiting_for_gps = False
            
            
            #self.active_acq_waiting_for_gps = False
            #else:
            #    self.previous_gps_msg_for_acq = msg
            #    self.previous_latency_from_gps_to_exposure = self.ros_time_for_most_recent_trigger - Time.from_msg(msg.header.stamp)
        
    def trigger(self):
        # Continually check until trigger is armed. Once the trigger is
        # armed, it is ready to be executed.
        self.current_image_msg_to_publish = ImageWithMetadata()
        
        while not self.device.nodemap.get_node('TriggerArmed').value:
            continue

        # Trigger an image buffer manually, since trigger mode is enabled.
        # This triggers the camera to acquire a single image buffer.
        # A buffer is then filled and moved to the output queue, where
        # it will wait to be retrieved.
        # Before the image buffer is sent, the exposure end event will
        # occur. This will happen on every iteration
        #self.device.nodemap.get_node('UserOutputValue').value = True
        #time.sleep(0.05)
        #libc.usleep(1000)
        try:
            self.device.nodemap.get_node('TriggerSoftware').execute()
        except:
            print("triggering failed..")
            return -1
        self.active_acq_waiting_for_gps = True
        #time.sleep(0.05)
        #libc.usleep(1000)
        #self.device.nodemap.get_node('UserOutputValue').value = False

        # Wait on the event to process it, invoking the registered callback.
        # The data is created from the event generation, not from waiting
        # on it. If the exposure time is long a Timeout exception may
        # occur unless large timeout value is
        # passed to the 'Device.wait_on_event()'
        #self.device.wait_on_event()
        
        try:
            self.device.wait_on_event(timeout=100)  # exposure start
        except:
            pass
        
        #print(datetime.now())
        ##################################
        ### Important observation ########
        ##################################
        # We can spend the time while waiting on the image transfer on other tasks. It is gettings transferred asap none the less
        #while(self.active_acq_waiting_for_gps):
        #    time.sleep(0.03)
        #    pass
        #time.sleep(0.2)
        print("receiving buffer..")
        try:
            image_buffer = self.device.get_buffer(timeout=500)
        except:
            print("Error retreiving image..")
            return -1
        chunk_node_names = ['ChunkExposureTime', 'ChunkGain']
        print("got chunk names")                             
        
        try:
            chunk_nodes_dict = image_buffer.get_chunk(
              chunk_node_names)

            # Print the value of the chunks for the current buexposure_start_timeffer
            self.current_image_msg_to_publish.exposure_time = chunk_nodes_dict['ChunkExposureTime'].value
            self.current_image_msg_to_publish.gain = chunk_nodes_dict['ChunkGain'].value
        except ValueError:
            print(f'\t\t\tFailed to get chunks')
                                     
        print("finished getting buffer..")
        #print(datetime.now())
        #print(type(image_buffer)) 
        image_np_array = np.ctypeslib.as_array( image_buffer.pdata,
		(image_buffer.height, image_buffer.width))
        self.current_image_msg_to_publish.image = self.bridge.cv2_to_imgmsg(image_np_array, 'mono8')

        # --------
        self.current_image_msg_to_publish.image.header.stamp = self.ros_time_for_most_recent_trigger.to_msg()
        self.current_image_msg_to_publish.header.stamp = self.ros_time_for_most_recent_trigger.to_msg()
        # --------
        self.current_image_msg_to_publish.image_number = self.image_number
        self.current_image_msg_to_publish.camera_height = self.current_camera_height
        #time.sleep(0.05)
        #nodes = self.device.nodemap.get_node(['ExposureTime', 'Gain'])
        #self.current_image_msg_to_publish.exposure_time = nodes["ExposureTime"].value
        #self.current_image_msg_to_publish.gain = nodes["Gain"].value
        if(self.active_acq_waiting_for_gps == True):
            print("gps was not ready in time....")
            self.active_acq_waiting_for_gps = False
        print("publishing image..")
        self.image_publisher.publish(self.current_image_msg_to_publish)
        
        self.device.requeue_buffer(image_buffer)
                    
   

def main(args=None):
    rclpy.init(args=args)

    global camera_node
    camera_node = camera_front_node()
    camera_node.device.start_stream()
    
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    
    #camera_node.run()
    #example_entry_point()
    #rclpy.spin(camera_node)
    
    try:
        camera_node.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        camera_node.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
