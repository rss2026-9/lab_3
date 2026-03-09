#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
import time

from wall_follower.visualization_tools import VisualizationTools
import math

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
        self.kp = 5.0
        self.kd = 2.0
        self.ki = 0.0
        self.front_thresh = 1.8
        self.integral = 0
        self.prev_error = 0.0
        self.prev_t = self.get_clock().now().nanoseconds * 1e-9
        self.latest_steer = 0
        self.mode = "follow"
        self.seek_trigger_dist = 2.0
        self.seek_steer = 0.8

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
        self.wall_pub = self.create_publisher(Marker, "/wall", 1)
        self.sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)
        self.timer = self.create_timer(0.05, self.drive_callback)
        self.get_logger().info(f"Subscribing to {self.SCAN_TOPIC}, publishing to {self.DRIVE_TOPIC}")


        # TODO: Write your callback functions here
    def drive_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.VELOCITY
        msg.drive.steering_angle = float(self.latest_steer)
        self.drive_pub.publish(msg)
        
    def scan_callback(self, msg: LaserScan):
        n = len(msg.ranges)
        angles = msg.angle_min + msg.angle_increment * np.arange(n)
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        
        
        if self.SIDE == 1:
        #left side: 90 deg is about 1.57 rad
            a1, a2 = 0, 2
        else:
        #right side
            a1, a2 = -2, -1
        #plot line for detected line
        pts = self.sector_points_xy(
            angles, ranges, a1, a2,
            range_min=msg.range_min,
            range_max=msg.range_max)
        self.plot_fitted_wall(pts, frame="/laser", half_length=2.0)
        #front angle limits
        front_a1, front_a2 = -0.35, 0.35
        #filter angles
        in_sector = (angles >= a1) & (angles <= a2)
        sector_ranges = ranges[in_sector]
        front_mask = (angles >= front_a1) & (angles <= front_a2)
        front_ranges = ranges[front_mask]
        front_ranges = front_ranges[np.isfinite(front_ranges)]
        #get general distances estimate to wall
        p0, d = self.fit_line_tls(pts)
        nvec = np.array([-d[1], d[0]])
        measured_dist = abs(float(p0 @ nvec))
        front_dist = float(np.median(front_ranges)) if front_ranges.size > 0 else float("inf")
        if front_dist < self.front_thresh:
            self.latest_steer = float(-1.0 * self.SIDE * self.seek_steer)
            self.integral = 0.0
            self.prev_error = 0.0
            return
        
        #follow case
        if self.mode == "follow":
            if measured_dist >= self.seek_trigger_dist:
                self.mode = "seek"
            else:
                #error
                error = measured_dist - float(self.DESIRED_DISTANCE)
                #time step
                t = self.get_clock().now().nanoseconds * 1e-9
                dt = t - self.prev_t
                if dt <= 1e-4:
                    dt = 1e-4
                #calulating PID terms
                self.integral += error * dt
                derivative = (error - self.prev_error) / dt
                #calculating new steer
                u = self.kp * error + self.ki * self.integral + self.kd * derivative
                steer = float(self.SIDE * float(u))
                if not np.isfinite(steer):
                    steer = 0.0
                #set params
                self.latest_steer = float(steer)
                #update state
                self.prev_error = error
                self.prev_t = t
                #debug check
                self.get_logger().info(f"dist={measured_dist:.2f} err={error:.2f} steer={steer:.2f}")
        #case for seek mode
        elif self.mode == "seek":
            if measured_dist < self.seek_trigger_dist:
                self.mode = "follow"
                self.integral = 0.0
                self.prev_error = 0.0
            else:
                steer = self.SIDE * self.seek_steer
                self.latest_steer = steer
                self.get_logger().info(f"dist={measured_dist:.2f} steer={steer:.2f} state: seeking")
            
        

        
        
        
        
    def range_at_angle(self, scan: LaserScan, angle_rad: float) -> float:
        """Return range at a desired angle (rad) with bounds + inf/NaN handling."""
        # Clamp angle into scan limits
        a = max(scan.angle_min, min(scan.angle_max, angle_rad))
        i = int(round((a - scan.angle_min) / scan.angle_increment))
        i = max(0, min(len(scan.ranges) - 1, i))
        r = scan.ranges[i]

        # Handle invalid values
        if math.isinf(r) or math.isnan(r):
            return float('inf')
        return r
    def sector_points_xy(self, angles: np.ndarray, ranges: np.ndarray,
                     a1: float, a2: float,
                     range_min: float = 0.5, range_max: float = 10.0):
        mask = (angles >= a1) & (angles <= a2)
        r = ranges[mask]
        th = angles[mask]
        valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        r = r[valid]
        th = th[valid]

        x = r * np.cos(th)
        y = r * np.sin(th)

        pts = np.stack([x, y], axis=1)
        pts = pts[pts[:, 0] > 0.0]

        return pts
    def fit_line_tls(self, pts: np.ndarray):
        """
        Total least squares line fit (PCA).
        Returns (p0, d) where p0 is centroid and d is unit direction vector.
        """
        p0 = pts.mean(axis=0)
        X = pts - p0
        #covariance and principal direction
        C = (X.T @ X) / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        d = eigvecs[:, np.argmax(eigvals)]
        d = d / np.linalg.norm(d)
        return p0, d
    def plot_fitted_wall(self, pts: np.ndarray, frame="/laser", half_length=2.0):
        """Fit TLS line to pts and publish it as a line marker."""
        p0, d = self.fit_line_tls(pts)
        p1 = p0 - half_length * d
        p2 = p0 + half_length * d
        x = np.array([p1[0], p2[0]])
        y = np.array([p1[1], p2[1]])
        VisualizationTools.plot_line(x, y, self.wall_pub, frame=frame)
    
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
    
