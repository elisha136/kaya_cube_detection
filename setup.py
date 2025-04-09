from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'cube_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elisha',
    maintainer_email='elisha@todo.todo',
    description='Cube detection and tracking using YOLO, depth camera, and EKF in ROS2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'live_capture_node = cube_detection.live_capture_node:main',
            'cube_detection_node = cube_detection.cube_detection_node:main',
            'motion_planning_subscriber = cube_detection.motion_planning_subscriber:main',
        ],
    },
)
