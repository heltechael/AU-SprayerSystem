import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'sprayer_system'
    node_executable = 'sprayer_node'

    config_file = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'sprayer_config.yaml'
    )

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    sprayer_node = Node(
        package=package_name,
        executable=node_executable,
        name='sprayer_control_node', 
        output='screen',
        emulate_tty=True,
        parameters=[config_file],
    )

    return LaunchDescription([
        sprayer_node
    ])