#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class GoStraight(Node):
    def __init__(self):
        super().__init__("go_straight")
        self.declare_parameter("drive_topic", "/vesc/high_level/ackermann_cmd")
        drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 2.5
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)


def main():
    rclpy.init()
    node = GoStraight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
