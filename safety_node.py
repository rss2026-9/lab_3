#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyController(Node):
    def __init__(self):
        super().__init__("safety_controller")

        # Topics (parameterized so you can swap sim vs car mux topic)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("input_drive_topic", "/drive_raw")   # your normal controller output
        self.declare_parameter("output_drive_topic", "/drive")  # goes to mux input

        # TTC behavior params
        self.declare_parameter("ttc_threshold", 0.6)   # seconds
        self.declare_parameter("distance_buffer", 0.25)  # meters (car radius + margin)
        self.declare_parameter("min_valid_range", 0.05)
        self.declare_parameter("max_valid_range", 15.0)

        # Hysteresis / anti-chatter
        self.declare_parameter("release_ttc", 0.9)     # seconds (must be > ttc_threshold)
        self.declare_parameter("hold_time", 0.15)      # seconds to hold brake once triggered

        # Optional: also brake if very close regardless of TTC (helps at low speed)
        self.declare_parameter("hard_stop_range", 0.35)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.in_drive_topic = self.get_parameter("input_drive_topic").value
        self.out_drive_topic = self.get_parameter("output_drive_topic").value

        self.ttc_threshold = float(self.get_parameter("ttc_threshold").value)
        self.release_ttc = float(self.get_parameter("release_ttc").value)
        self.buffer = float(self.get_parameter("distance_buffer").value)
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

        self.latest_drive = AckermannDriveStamped()
        self.latest_speed = 0.0

        self.braking = False
        self.brake_until = 0.0
        self.latest_min_ttc = float("inf")
        self.latest_min_range = float("inf")

        self.get_logger().info(
            f"SafetyController: scan={self.scan_topic}, in_drive={self.in_drive_topic}, out_drive={self.out_drive_topic}"
        )

    def drive_cb(self, msg: AckermannDriveStamped):
        self.latest_drive = msg
        self.latest_speed = float(msg.drive.speed)

    def scan_cb(self, scan: LaserScan):
        # If we haven't received a drive cmd yet, do nothing (or publish 0 if you prefer)
        v = float(self.latest_speed)

        n = len(scan.ranges)
        if n == 0:
            return

        angles = scan.angle_min + scan.angle_increment * np.arange(n, dtype=np.float32)
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        # Valid ranges only
        valid = np.isfinite(ranges) & (ranges >= self.rmin) & (ranges <= self.rmax)
        if not np.any(valid):
            return

        angles = angles[valid]
        ranges = ranges[valid]

        # Consider only beams that are in the forward-ish half-plane.
        # TTC uses projection v*cos(theta). If cos(theta) <= 0, you're moving away from that beam direction.
        cos_th = np.cos(angles)
        v_parallel = v * cos_th

        forward = v_parallel > 1e-3  # only beams where we're moving toward obstacle
        min_ttc = float("inf")

        if np.any(forward) and v > 1e-3:
            r_f = ranges[forward]
            vp_f = v_parallel[forward]

            # TTC = (range - buffer) / v_parallel
            ttc = (r_f - self.buffer) / vp_f
            # only meaningful TTC where range > buffer
            ttc = ttc[np.isfinite(ttc)]
            if ttc.size > 0:
                min_ttc = float(np.min(ttc))

        min_range = float(np.min(ranges)) if ranges.size > 0 else float("inf")

        self.latest_min_ttc = min_ttc
        self.latest_min_range = min_range

        now = self.get_clock().now().nanoseconds * 1e-9

        # Trigger conditions
        trigger = (min_ttc < self.ttc_threshold) or (min_range < self.hard_stop_range)

        # Hysteresis logic
        if trigger:
            self.braking = True
            self.brake_until = max(self.brake_until, now + self.hold_time)
        else:
            # Only release if:
            # 1) hold time expired
            # 2) TTC has recovered above release threshold (or we don't have TTC)
            if self.braking and now >= self.brake_until:
                if (min_ttc >= self.release_ttc) or (min_ttc == float("inf")):
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
                f"BRAKE: min_ttc={self.latest_min_ttc:.2f} min_range={self.latest_min_range:.2f} v={v:.2f}"
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

