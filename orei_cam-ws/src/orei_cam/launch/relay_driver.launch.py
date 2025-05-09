from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    
   
            
    #serial_driver_node = IncludeLaunchDescription(
    #            PythonLaunchDescriptionSource([
    #                FindPackageShare("serial_driver"), '/launch', '/serial_driver_bridge_node.launch.py'])
    #        )
            
    
            
    relay_driver_node = Node(
                package='relay_driver',
                executable='relay_driver',
                name='relay_driver',
                respawn=True,
                output='screen'
                )
                
    return LaunchDescription([
          relay_driver_node
          ])
