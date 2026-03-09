#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64


class SafetyController(Node):
    def __init__(self):
        super().__init__("safety_controller")

        # Topics (parameterized so you can swap sim vs car mux topic)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("input_drive_topic", "/vesc/high_level/ackermann_cmd")   # your normal controller output
        self.declare_parameter("output_drive_topic", "/vesc/low_level/input/safety")  # goes to mux input
        self.declare_parameter("safety_error_topic", "/safety_error")

        self.declare_parameter("min_valid_range", 0.05)
        self.declare_parameter("max_valid_range", 15.0)
        self.declare_parameter("hold_time", 0.15)      # seconds to hold brake once triggered
        self.declare_parameter("hard_stop_range", 1.3)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.in_drive_topic = self.get_parameter("input_drive_topic").value
        self.out_drive_topic = self.get_parameter("output_drive_topic").value
        self.safety_error_topic = self.get_parameter("safety_error_topic").value

        self.rmin = float(self.get_parameter("min_valid_range").value)
        self.rmax = float(self.get_parameter("max_valid_range").value)
        self.hard_stop_range = float(self.get_parameter("hard_stop_range").value)
        self.hold_time = float(self.get_parameter("hold_time").value)

        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, self.in_drive_topic, self.drive_cb, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_cb, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.out_drive_topic, 10
        )
        self.safety_error_pub = self.create_publisher(Float64, self.safety_error_topic, 10)

        self.latest_drive = AckermannDriveStamped()
        self.latest_speed = 0.0

        self.braking = False
        self.brake_until = 0.0
        self.latest_min_range = float("inf")

        self.get_logger().info(
            f"SafetyController: scan={self.scan_topic}, in_drive={self.in_drive_topic}, "
            f"out_drive={self.out_drive_topic}, safety_error={self.safety_error_topic}"
        )

    def drive_cb(self, msg: AckermannDriveStamped):
        self.latest_drive = msg
        self.latest_speed = float(msg.drive.speed)

    def scan_cb(self, scan: LaserScan):
        n = len(scan.ranges)
        if n == 0:
            return

        angles = scan.angle_min + scan.angle_increment * np.arange(n, dtype=np.float32)
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        # Restrict to -30 deg to +30 deg cone in front
        angle_mask = (angles >= -np.pi / 6) & (angles <= np.pi / 6)

        # Valid ranges only
        valid = np.isfinite(ranges) & (ranges >= self.rmin) & (ranges <= self.rmax)
        mask = angle_mask & valid
        if not np.any(mask):
            return

        ranges = ranges[mask]
        min_range = float(np.min(ranges))
        self.latest_min_range = min_range

        # safety_error = hard_stop_range - min_range (positive = too close)
        safety_error = self.hard_stop_range - min_range
        err_msg = Float64()
        err_msg.data = safety_error
        self.safety_error_pub.publish(err_msg)

        now = self.get_clock().now().nanoseconds * 1e-9

        # Trigger: brake when min_range < hard_stop_range
        trigger = min_range < self.hard_stop_range

        # Hysteresis: hold brake for hold_time, then release when min_range >= hard_stop_range
        if trigger:
            self.braking = True
            self.brake_until = max(self.brake_until, now + self.hold_time)
        elif self.braking and now >= self.brake_until and min_range >= self.hard_stop_range:
            self.braking = False

        # Publish either passthrough or overridden command
        out = AckermannDriveStamped()
        out.header = self.latest_drive.header
        out.drive = self.latest_drive.drive  # copy command

        if self.braking:
            out.drive.speed = 0.0
            # Optionally keep steering to “turn away” while braking:
            # out.drive.steering_angle = self.latest_drive.drive.steering_angle
            # Or reduce steering to avoid spin:
            # out.drive.steering_angle *= 0.5

            self.get_logger().info(
                f"BRAKE: min_range={min_range:.2f} safety_error={safety_error:.2f}"
            )

        self.drive_pub.publish(out)


def main():
    rclpy.init()
    node = SafetyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

