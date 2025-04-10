from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cube_detection',
            executable='live_capture_node',  # Correct executable name
            name='live_capture_node',
            output='screen'
        ),
        Node(
            package='cube_detection',
            executable='cube_detection_node',  # Correct executable name
            name='cube_detection_node',
            output='screen'
        ),
        Node(
            package='cube_detection',
            executable='motion_planning_subscriber',  # Correct executable name
            name='motion_planning_subscriber',
            output='screen'
        )
    ])