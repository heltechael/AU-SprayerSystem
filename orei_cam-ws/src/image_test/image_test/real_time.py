from rosidl_runtime_py import message_to_ordereddict
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from orei_cam_interfaces.msg import ImageWithMetadata
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import csv
from datetime import datetime

class ImageTest(Node):
    def __init__(self):
        super().__init__('image_test')

        # Load YOLOv8 TensorRT model
        self.model_path = "/media/OREI-cam/orei-micro-dose-sprayer/models/instance_segmentation_v_0.3/yolov8x-seg/weights/yolov8x_seg_fp16_1664x960.engine"
        self.model = YOLO(self.model_path, task="segment")

        # Create a subscription for images
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            ImageWithMetadata,
            'images',
            self.image_received_callback,
            10)

        self.image_data = ImageWithMetadata()
        self.get_logger().info('ImageTest node has been initialized.')

        # Flag to indicate if the node is busy processing an image
        self.processing = False

        # Image counter
        self.image_counter = 0

        # Initialize CSV file and buffer for every 5 images
        self.csv_file = '/home/orei/image_test_output/detections.csv'
        self.buffer = []  # Buffer to store CSV rows before writing
        self.buffer_size = 5  # Number of images before writing to the CSV

        # Initialize CSV file with all metadata fields including GPS position
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row including metadata fields
            writer.writerow(['imgname', 'DateTime', 'ImageArea', 'GPS_latitude', 'GPS_longitude', 'GPS_altitude', 'imgheight', 'imgwidth', 'frame_id', 'timestamp_sec', 'timestamp_nanosec',
                             'Monocot weed', 'Dicot weed', 'Bean', 'Lupin', 'Pea',
                             'Buck wheat', 'Beet', 'Maize', 'Cereal', 'Unknown', 'Full_Metadata'])

    def process_image(self):
        # Convert Bayer to RGB and resize the image to 1664x960
        colour_image_gamma_adjusted = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.image_data.image), cv2.COLOR_BAYER_RG2RGB)
        colour_image_gamma_adjusted = cv2.resize(colour_image_gamma_adjusted, (1664, 960))  # Resizing for inference
        return colour_image_gamma_adjusted

    def image_received_callback(self, msg):
        # Check if the node is already processing an image
        if self.processing:
            # Skip this image because the system is busy
            self.get_logger().info('Skipping image as processing is still ongoing.')
            return

        # Set the flag to indicate that processing has started
        self.processing = True

        self.get_logger().info('Received an image!')

        # Store the image data for processing
        self.image_data = msg

        # Convert the message to an ordered dictionary to extract all metadata
        metadata_dict = message_to_ordereddict(msg)

        # Extract GPS and other metadata directly from the dictionary
        gps_latitude = metadata_dict.get('gpsfix', {}).get('latitude', 'N/A')
        gps_longitude = metadata_dict.get('gpsfix', {}).get('longitude', 'N/A')
        gps_altitude = metadata_dict.get('gpsfix', {}).get('altitude', 'N/A')
        imgheight = msg.image.height
        imgwidth = msg.image.width
        frame_id = msg.image.header.frame_id
        timestamp_sec = msg.image.header.stamp.sec
        timestamp_nanosec = msg.image.header.stamp.nanosec

        # Process the image (color correction and resizing)
        processed_image = self.process_image()

        # Run YOLOv8 inference on the processed image (using the resized 1664x960 size)
        results = self.model.predict(source=processed_image, imgsz=(1664, 960))

        # Initialize the count for each class (0-9)
        class_counts = {i: 0 for i in range(10)}  # Dictionary to count detections for each class

        # Count the detections by class
        for detection in results[0].boxes:
            class_id = int(detection.cls)  # Get the class ID
            class_counts[class_id] += 1    # Increment the count for the detected class

        # Get the total number of detected objects and log it
        num_detections = sum(class_counts.values())
        self.get_logger().info(f'Number of objects detected: {num_detections}')

        # Prepare the data for the CSV
        imgname = f'processed_image_{self.image_counter}.png'
        datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_area = 1664 * 960  # Image area based on resolution

        # Add the data to the buffer, including GPS position and metadata
        self.buffer.append([imgname, datetime_now, image_area, gps_latitude, gps_longitude, gps_altitude, imgheight, imgwidth, frame_id, timestamp_sec, timestamp_nanosec,
                            class_counts[0], class_counts[1], class_counts[2], class_counts[3],
                            class_counts[4], class_counts[5], class_counts[6], class_counts[7],
                            class_counts[8], class_counts[9], str(metadata_dict)])

        # Write to CSV after every 5 images
        if len(self.buffer) >= self.buffer_size:
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.buffer)
            self.buffer.clear()  # Clear the buffer after writing

        self.get_logger().info(f'Buffered detection data for {self.image_counter + 1} image(s)')

        # Increment the image counter for unique filenames
        self.image_counter += 1

        # Reset the flag to allow processing of the next image
        self.processing = False

def main(args=None):
    rclpy.init(args=args)

    # Initialize the ImageTest node
    image_test = ImageTest()

    # Spin the node to process incoming images one by one
    rclpy.spin(image_test)

    # Ensure the node is properly destroyed when shutting down
    image_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


