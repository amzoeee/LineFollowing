#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from line_interfaces.msg import Line
import tf_transformations as tft

#############
# CONSTANTS #
#############
_RATE = 10 # (Hz) rate for rospy.rate
_MAX_SPEED = 2.0 # (m/s)
_MAX_CLIMB_RATE = 1.0 # m/s
_MAX_ROTATION_RATE = 5.0 # rad/s 
IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = 300 # Number of pixels forward to extrapolate the line

KP_X = 0.004
KP_Y = 0.004
KP_Z_W = 0.5  # Proportional gains for x, y, and angular velocity control

KD_X = 0.002
KD_Y = 0.002
KD_Z_W = 0.05

DISPLAY = True

class LineController(Node):
    def __init__(self) -> None:
        super().__init__('line_controller')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.line_sub = self.create_subscription(
            Line, '/line/param', self.line_sub_cb, 1)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -2.0

        # Linear setpoint velocities in downward camera frame
        self.vx__dc = 0.0
        self.vy__dc = 0.0
        self.vz__dc = 0.0

        self.prev_x_error = 0.0
        self.prev_y_error = 0.0
        self.prev_w_error = 0.0

        # Yaw setpoint velocities in downward camera frame
        self.wz__dc = 0.0

        # Quaternion representing the rotation of the drone's body frame in the NED frame. initiallize to identity quaternion
        self.quat_bu_lenu = (0, 0, 0, 1)

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx: float, vy: float, wz: float) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [None, None, self.takeoff_height]
        if self.offboard_setpoint_counter < 100:
            msg.velocity = [0.0, 0.0, 0.0]
        else:
            msg.velocity = [vx, vy, 0.0] # wants lenu ???
        msg.acceleration = [None, None, None]
        msg.yaw = float('nan')
        msg.yawspeed = wz
        # msg.yawspeed = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing velocity setpoints {[vx, vy, wz]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def convert_velocity_setpoints(self):
        '''Convert velocity setpoints from downward camera frame to lenu frame'''
        vx, vy, vz = self.dc2lned((self.vx__dc, self.vy__dc, self.vz__dc))
        _, _, wz = self.dc2lned((0.0, 0.0, self.wz__dc))

        # enforce safe velocity limits
        if _MAX_SPEED < 0.0 or _MAX_CLIMB_RATE < 0.0 or _MAX_ROTATION_RATE < 0.0:
            raise Exception("_MAX_SPEED,_MAX_CLIMB_RATE, and _MAX_ROTATION_RATE must be positive")
        vx = min(max(vx,-_MAX_SPEED), _MAX_SPEED)
        vy = min(max(vy,-_MAX_SPEED), _MAX_SPEED)
        wz = min(max(wz,-_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        return (vx, vy, wz)

    def dc2lned(self, vector):
        '''Use current yaw to convert vector from downward camera frame to lenu frame'''
        v4 = np.array([[vector[0]],
                        [vector[1]],
                        [vector[2]],
                        [     0.0]])
        
        yaw = self.vehicle_local_position.heading
        self.get_logger().info("yaw: " + str(yaw))

        R_dc2lned = np.array([[-np.sin(yaw), np.cos(yaw), 0.0, 0.0], 
                                 [np.cos(yaw), np.sin(yaw), 0.0, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]]) 

        output = np.dot(R_dc2lned, v4)
        
        return (output[0,0], output[1,0], output[2,0])
        
    
    def timer_callback(self) -> None:
        """Callback function for the timer."""

        self.publish_offboard_control_heartbeat_signal()
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        self.offboard_setpoint_counter += 1
    
    def line_sub_cb(self, param):
        """
        Callback function which is called when a new message of type Line is recieved by self.line_sub.
        Notes:
        - This is the function that maps a detected line into a velocity 
        command
            
            Args:
                - param: parameters that define the center and direction of detected line
        """
        # self.get_logger().info("Following line")
        # Extract line parameters
        x, y, vx, vy = param.x, param.y, param.vx, param.vy
        line_point = np.array([x, y])
        line_dir = np.array([vx, vy])
        line_dir = line_dir / np.linalg.norm(line_dir)  # Ensure unit vector

        if line_dir[1] < 0:
            line_dir = -line_dir

        # Target point EXTEND pixels ahead along the line direction
        target = line_point + EXTEND * line_dir

        # Error between center and target
        error = target - CENTER

        # Set linear velocities (downward camera frame)
        self.vx__dc = KP_X * error[0]
        self.vy__dc = KP_Y * error[1]

        self.vx__dc += KD_X * (error[0]-self.prev_x_error)/0.1
        self.vy__dc += KD_Y * (error[1]-self.prev_y_error)/0.1

        self.prev_x_error = error[0]
        self.prev_y_error = error[1]

        # Get angle between y-axis and line direction
        # Positive angle is counter-clockwise
        forward = np.array([0.0, 1.0])
        angle_error = math.atan2(-line_dir[0], line_dir[1])

        # Set angular velocity (yaw)
        self.wz__dc = KP_Z_W * angle_error

        self.wz__dc += KD_Z_W * (angle_error-self.prev_w_error)/0.1

        self.prev_w_error = angle_error

        self.publish_trajectory_setpoint(*self.convert_velocity_setpoints())

        self.get_logger().info(f"x error: {error[0]}, y error: {error[1]}, angle error: {angle_error}")

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = LineController()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)