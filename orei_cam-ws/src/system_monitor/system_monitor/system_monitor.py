import rclpy
from rclpy.node import Node
import psutil
import subprocess
import os
from datetime import datetime
from std_msgs.msg import Empty
from rcl_interfaces.msg import Log
from rclpy.qos import qos_profile_sensor_data

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')
        
        self.log_dir = '/home/orei/run_logs/'
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file_path = os.path.join(self.log_dir, log_file_name)
        
        self.trigger_count = 0
        
        self.expected_nodes = [
            '/camera_height',
            '/orei_cam_gui',
            '/septentrio_gnss_driver',
            '/static_transform_publisher',
            '/system_monitor',
            '/yolov8_seg_node',
            # Add any other expected nodes here
        ]
        
        self.trigger_subscriber = self.create_subscription(
            Empty,
            '/trigger',
            self.trigger_callback,
            qos_profile_sensor_data
        )
        
        self.rosout_subscriber = self.create_subscription(
            Log,
            '/rosout',
            self.rosout_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.5, self.log_system_info)
    
    def log_system_info(self):
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        log_message = f"CPU Usage: {cpu_usage}%\nMemory Usage: {memory_info.percent}%\n"
        log_message += f"Number of messages published to /trigger in the last interval: {self.trigger_count}\n"
        self.trigger_count = 0
        
        running_nodes = self.get_ros2_nodes()
        if running_nodes:
            log_message += f"Nodes running: {', '.join(running_nodes)}\n"
            not_running_nodes = self.get_not_running_nodes(running_nodes)
            if not_running_nodes:
                log_message += f"Nodes not running: {', '.join(not_running_nodes)}\n"
            else:
                log_message += "All expected nodes are running.\n"
        else:
            log_message += "No nodes found\n"
        
        self.write_log(log_message)

    def get_ros2_nodes(self):
        try:
            node_list = subprocess.check_output(["ros2", "node", "list"]).decode().splitlines()
            return node_list
        except subprocess.CalledProcessError as e:
            self.write_log(f"Failed to get node list: {e}\n")
            return None

    def get_not_running_nodes(self, running_nodes):
        not_running = [node for node in self.expected_nodes if node not in running_nodes]
        return not_running

    def trigger_callback(self, msg):
        self.trigger_count += 1

    def rosout_callback(self, msg):
        self.write_log(f"ROS2 Log Message: {msg.msg}")

    def write_log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

        self.get_logger().info(message)

def main(args=None):
    rclpy.init(args=args)
    system_monitor = SystemMonitor()
    rclpy.spin(system_monitor)
    
    system_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

