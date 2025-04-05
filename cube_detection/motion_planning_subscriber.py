#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class MotionPlanningSubscriber(Node):
    def __init__(self):
        super().__init__('motion_planning_subscriber')
        
        # Subscriber to the cube's position
        self.subscription = self.create_subscription(
            Point,
            '/cube/position',
            self.position_callback,
            10
        )
        
        # Publisher to forward the cube's position
        self.publisher = self.create_publisher(
            Point,
            '/motion_planning/cube_position',
            10
        )

    def position_callback(self, msg):
        # Log the received cube position
        self.get_logger().info(f"Received cube position: x={msg.x}, y={msg.y}, z={msg.z}")
        
        # Forward the cube's position to the motion planning node
        self.publisher.publish(msg)
        self.get_logger().info(f"Forwarded cube position to /motion_planning/cube_position")

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()