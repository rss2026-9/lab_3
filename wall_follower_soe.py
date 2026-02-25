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
  
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            10
        )

        # Subscriber: receive LiDAR scans
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10
        )

    def scan_callback(self, scan: LaserScan):
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=scan.range_max,
            posinf=scan.range_max,
            neginf=scan.range_max
        )
        n = len(ranges)
        if n == 0:
            return

        s = 1 if self.SIDE == 1 else -1


        theta = np.deg2rad(50.0)
        angle_b = s * np.deg2rad(110)            # side
        angle_a = s * np.deg2rad(40)     # 40 deg

        def idx(angle):
            i = int(round((angle - scan.angle_min) / scan.angle_increment))
            return int(np.clip(i, 0, n - 1))

        def window_ranges(center_angle, half_width_deg):
            half = np.deg2rad(half_width_deg)
            i0 = idx(center_angle - half)
            i1 = idx(center_angle + half)
            if i0 > i1:
                i0, i1 = i1, i0
            return ranges[i0:i1+1]

        SIDE_WIN_DEG = 6.0      
        A_WIN_DEG = 6.0         
        FRONT_WIN_DEG = 10.0    

        a_win = window_ranges(angle_a, A_WIN_DEG)
        b_win = window_ranges(angle_b, SIDE_WIN_DEG)
        front_win = window_ranges(0.0, FRONT_WIN_DEG)

        
        a = float(np.median(a_win))
        b = float(np.median(b_win))
        front = float(np.min(front_win))
        if a > 8.0 and b > 8.0:
            steer = 0.2 * s      
            speed = 1
        else:
           
            alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
            dist = b * np.cos(alpha)

            error = dist - self.DESIRED_DISTANCE


            kP = 10
            kD = 2

            if not hasattr(self, "prev_error"):
                self.prev_error = 0.0
            if not hasattr(self, "integral"):
                self.integral = 0.0

            deriv = error - self.prev_error
            self.prev_error = error


            steer = s * (kP * error + kD * deriv)

            
            steer = float(np.clip(steer, -0.34, 0.34))

            speed = float(self.VELOCITY)

        
        if front < 0.25:
         speed = 0.0
         steer = 0.0

        cmd = AckermannDriveStamped()
        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steer)
        self.drive_pub.publish(cmd)   
    
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
    
