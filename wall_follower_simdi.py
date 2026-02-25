#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from wall_follower.visualization_tools import VisualizationTools


def clamp(x, lo, hi):
    """For limiting values ie steering, or scan data"""
    return max(lo, min(hi, x))


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

        # Pub/sub
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10) #Creates a publisher that sends AckermannDriveStamped messages to drive_topic
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10) #Subscribes to the lidar topic. Every scan triggers scan_callback

        self.viz = None

        ###########################
        # Controller configuration
        ###########################
        # Simulator max steering angle
        self.MAX_STEER = 0.34

        # PD gains (tune if needed)
        self.KP = 1.2
        self.KD = 0.35

        # Lookahead (meters) for corner anticipation
        self.LOOKAHEAD = 1.0

        # Derivative filter (0..1): higher = smoother D term
        self.D_FILTER = 0.85 #smoothing factor (closer to 1 = smoother)
        self.prev_err = None #previous error and time to get derivative
        self.prev_t = None #filtered derivative (smooths noise)
        self.d_filt = 0.0

    def scan_callback(self, msg: LaserScan):
        t_now = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec #converts ROS time stamp into seconds as a float. Used for derivative d(err)/dt

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf) #replaces invalid array values with inf

        # range at a given angle using a small median window
        def range_at(angle_rad: float) -> float:
            """helper that returns a stable distance at a requested angle"""
            ang = clamp(angle_rad, msg.angle_min, msg.angle_max) #make sure the requested angle is within the lidar scan’s angular range
            idx = int(round((ang - msg.angle_min) / msg.angle_increment))
            idx = int(clamp(idx, 0, len(ranges) - 1))

            w = 3
            lo = int(clamp(idx - w, 0, len(ranges) - 1))
            hi = int(clamp(idx + w, 0, len(ranges) - 1))
            val = float(np.median(ranges[lo:hi + 1])) #takes a median over a small window of rays around that angle to avoid outliers

            if not np.isfinite(val):
                val = msg.range_max
            return float(clamp(val, msg.range_min, msg.range_max)) #if still invalid, treat as “far away”

        # Two rays on the chosen side:
        theta = np.deg2rad(50.0)      # forward-ish
        side_ang = np.deg2rad(90.0)   # directly to the side

        #for left wall, use positive angles, vice versa
        #a and b are the two distances used for the wall geometry
        if self.SIDE >= 0:  # left wall
            a = range_at(+theta)
            b = range_at(+side_ang)
        else:               # right wall
            a = range_at(-theta)
            b = range_at(-side_ang)

        # if wall not visible on that side, go straight
        if (not np.isfinite(a)) or (not np.isfinite(b)) or b >= 0.95 * msg.range_max:
            self._reset_state()
            self._publish_drive(0.0)
            return

        # Wall angle alpha
        denom = a * np.sin(theta)
        if denom < 1e-6:
            self._reset_state()
            self._publish_drive(0.0)
            return

        alpha = np.arctan2((a * np.cos(theta) - b), denom)

        # perpendicular distance to wall now
        d = b * np.cos(alpha)

        # predict distance ahead to anticipate corners
        d_future = d + float(self.LOOKAHEAD) * np.sin(alpha)

        # self.DESIRED_DISTANCE from params.yaml
        err = float(self.DESIRED_DISTANCE - d_future) #if d_future is smaller than desired (too close), err is positive.

        # Derivative term (filtered)
        derr = 0.0
        if self.prev_t is not None and self.prev_err is not None:
            dt = float(t_now - self.prev_t)
            if dt > 1e-3:
                raw = (err - self.prev_err) / dt #computing derivative
                self.d_filt = self.D_FILTER * self.d_filt + (1.0 - self.D_FILTER) * raw
                derr = float(self.d_filt)

        self.prev_t = t_now
        self.prev_err = err

        # PD control
        steer = float(self.KP * err + self.KD * derr)


        steer = (-self.SIDE) * steer #if SIDE = +1 (left wall): multiply by -1 so “too close” produces a right turn.


        steer = clamp(steer, -self.MAX_STEER, self.MAX_STEER) #ever command beyond the simulator’s max steering

        self._publish_drive(steer)

    def _publish_drive(self, steering_angle: float):
        drive = AckermannDriveStamped()
        #self.VELOCITY (from params.yaml)
        drive.drive.speed = float(self.VELOCITY)
        drive.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive)

    def _reset_state(self):
        self.prev_err = None
        self.prev_t = None
        self.d_filt = 0.0

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
