from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    
    camera_node_front = Node(
                package='camera_node_front',
                executable='camera_node_front',
                name='camera_node_front',
                output='screen',
                respawn=True
                )
                
    
    gps_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare("septentrio_gnss_driver"), '/launch', '/rover.launch.py'])#,
                #launch_arguments={
                #'file_name': 'rover_node.yaml'}
            )
            
    serial_driver_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare("serial_driver"), '/launch', '/serial_driver_bridge_node.launch.py'])
            )
            
    camera_trigger_node = Node(
                package='camera_trigger',
                executable='camera_trigger',
                name='camera_trigger',
                respawn=True
                )
            
    image_uploader_node = Node(
                package='image_uploader',
                executable='image_uploader',
                name='image_uploader',
                respawn=True
                )
            
    spraying_state_control_node = Node(
                package='spraying_state_control',
                executable='spraying_state_control',
                name='spraying_state_control',
                respawn=True
                )
            
    orei_cam_gui_node = Node(
                package='orei_cam_gui',
                executable='orei_cam_gui',
                name='orei_cam_gui',
                respawn=True
                )
            
    lidar_driver_node = Node(
                package='lidar_driver',
                executable='lidar_driver',
                name='lidar_driver',
                respawn=True
                )
            
    camera_height_node = Node(
                package='camera_height',
                executable='camera_height',
                name='camera_height',
                respawn=True
                )
            
    image_writer_node = Node(
                package='image_writer',
                executable='image_writer',
                name='image_writer',
                output='screen',
                respawn=True
                )
                
    return LaunchDescription([
          camera_node_front,
          gps_node,
          serial_driver_node,
          camera_trigger_node,
          spraying_state_control_node,
          #image_uploader_node,
          #orei_cam_gui_node,
          lidar_driver_node,
          camera_height_node,
          #image_writer_node
          ])
