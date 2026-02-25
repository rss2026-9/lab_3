#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS!
        self.add_on_set_parameters_callback(self.parameters_callback)

        # TODO: Initialize your publishers and subscribers here
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)

        self.kp = 1.15
        self.kd = 0.2
        self.max_steer = 0.38
        self.lookahead_x = 1.5

        self.prev_error = 0.0
        self.prev_time = None

        self.viz = VisualizationTools()

        # TODO: Write your callback functions here
    def scan_callback(self, scan: LaserScan):
        now = self.get_clock().now()
        if self.prev_time is None:
            self.prev_time = now
            return
        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now
        if dt <= 1e-6:
            return

        n = len(scan.ranges)
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment

        deg_min = np.deg2rad(20.0)
        deg_max = np.deg2rad(100.0)

        if self.SIDE == 1:
            mask = (angles >= deg_min) & (angles <= deg_max)
        else:
            mask = (angles <= -deg_min) & (angles >= -deg_max)

        mask &= np.isfinite(ranges) & (ranges > 0.05) & (ranges < 2.5)

        if np.count_nonzero(mask) < 20:
            deg_max_recovery = np.deg2rad(150.0)
            if self.SIDE == 1:
                mask = (angles >= deg_min) & (angles <= deg_max_recovery)
            else:
                mask = (angles <= -deg_min) & (angles >= -deg_max_recovery)
            mask &= np.isfinite(ranges) & (ranges > 0.05) & (ranges < 7.0)

        if np.count_nonzero(mask) < 10:
            self.publish_drive(self.VELOCITY, 0.05 * self.SIDE)
            return

        sel_r = ranges[mask]
        sel_a = angles[mask]

        xs = sel_r * np.cos(sel_a)
        ys = sel_r * np.sin(sel_a)

        front_mask = xs > -0.5
        xs = xs[front_mask]
        ys = ys[front_mask]

        if xs.shape[0] < 10:
            self.publish_drive(self.VELOCITY, 0.05 * self.SIDE)
            return

        # fit
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        sol, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        m = np.clip(float(sol[0]), -3.0, 3.0)
        b = float(sol[1])

        lookahead = 1.2
        y_wall = m * lookahead + b
        y_wall = np.clip(y_wall, -5.0, 5.0)

        target_y = float(self.SIDE) * float(self.DESIRED_DISTANCE)
        error = y_wall - target_y

        derr = (error - self.prev_error) / dt
        self.prev_error = error

        steer = self.kp * error + self.kd * derr
        if steer > self.max_steer:
            steer = self.max_steer
        elif steer < -self.max_steer:
            steer = -self.max_steer

        # turn_ratio = abs(steer) / self.max_steer
        # scale = 1.0 - 0.6 * turn_ratio
        # if scale < 0.4:
        #     scale = 0.4
        # speed_cmd = self.VELOCITY * scale
        # self.publish_drive(speed_cmd, steer)
        self.publish_drive(self.VELOCITY, steer)

    def publish_drive(self, speed: float, steering_angle: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!

        This is used by the test cases to modify the parameters during testing.
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
