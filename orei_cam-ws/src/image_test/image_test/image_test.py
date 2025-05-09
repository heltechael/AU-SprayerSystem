import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from orei_cam_interfaces.msg import ImageWithMetadata
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ImageTest(Node):
    def __init__(self):
        super().__init__('image_test')
        
        self.model_path = "/media/OREI-cam/orei-micro-dose-sprayer/models/instance_segmentation_v_0.2/yolov8x-seg/weights/yolov8x_seg_fp16_1664x960.engine"
        self.model = YOLO(self.model_path, task="segment")
        
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            ImageWithMetadata,
            'images',
            self.image_received_callback,
            10)
        
        self.image_data = ImageWithMetadata()
        self.get_logger().info('ImageTest node has been initialized.')
        self.image_counter = 0

    def process_image(self):
        colour_image_gamma_adjusted = cv2.cvtColor(self.bridge.imgmsg_to_cv2(self.image_data.image), cv2.COLOR_BAYER_RG2RGB)
        colour_image_gamma_adjusted = cv2.resize(colour_image_gamma_adjusted, (1664, 960))  # Updated size
        return colour_image_gamma_adjusted

    def image_received_callback(self, msg):
        self.get_logger().info('Received an image!')
        
        self.image_data = msg
        processed_image = self.process_image()
        results = self.model.predict(source=processed_image, imgsz=(1664, 960))  # Updated size
        num_detections = len(results[0].boxes)
        self.get_logger().info(f'Number of objects detected: {num_detections}')
        
        save_path = f"/home/orei/image_test_output/processed_image_{self.image_counter}.png"
        cv2.imwrite(save_path, processed_image)
        self.get_logger().info(f'Processed and saved image to {save_path}')
        
        self.image_counter += 1

def main(args=None):
    rclpy.init(args=args)
    image_test = ImageTest()
    rclpy.spin(image_test)
    image_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

