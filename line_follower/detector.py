#!/usr/bin/env python

###########
# IMPORTS #
###########
import numpy as np
import rclpy 
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from line_interfaces.msg import Line
import sys

#############
# CONSTANTS #
#############
LOW = np.array([0, 0, 0])  # Lower image thresholding bound
HI = np.array([255, 255, 255])   # Upper image thresholding bound
LENGTH_THRESH = None  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True

class LineDetector(Node):
    def __init__(self):
        super().__init__('detector')

        # A subscriber to the topic '/aero_downward_camera/image'
        self.camera_sub = self.create_subscription(
            Image,
            '/world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image',
            self.camera_sub_cb,
            10
        )

        # A publisher which will publish a parametrization of the detected line to the topic '/line/param'
        self.param_pub = self.create_publisher(Line, '/line/param', 1)

        # A publisher which will publish an image annotated with the detected line to the topic 'line/detector_image'
        self.detector_image_pub = self.create_publisher(Image, '/line/detector_image', 1)

        # Initialize instance of CvBridge to convert images between OpenCV images and ROS images
        self.bridge = CvBridge()

    ######################
    # CALLBACK FUNCTIONS #
    ######################
    def camera_sub_cb(self, msg):
        """
        Callback function which is called when a new message of type Image is received by self.camera_sub.
        """
        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Try to detect two perpendicular lines first
        two_lines = self.detect_two_lines(image)
        
        if two_lines is not None:
            # Two lines detected - combine them
            line1 = two_lines['line1']
            line2 = two_lines['line2']
            
            x1, y1, vx1, vy1 = line1
            x2, y2, vx2, vy2 = line2
            
            # Use average of the two line centers as the combined center point
            avg_x = (x1 + x2) / 2
            avg_y = (y1 + y2) / 2
            
            # Direction: average of both line directions (normalized)
            avg_vx = (vx1 + vx2) / 2
            avg_vy = (vy1 + vy2) / 2
            norm = np.sqrt(avg_vx**2 + avg_vy**2)
            avg_vx, avg_vy = avg_vx/norm, avg_vy/norm
            
            chosen_line = (avg_x, avg_y, avg_vx, avg_vy)
            
            # Publish the chosen line
            msg = Line()
            msg.x, msg.y, msg.vx, msg.vy = chosen_line
            self.param_pub.publish(msg)
            
            # Draw both lines for visualization
            if DISPLAY:
                annotated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Draw line 1 in red
                x, y, vx, vy = line1
                pt1 = (int(x - 100*vx), int(y - 100*vy))
                pt2 = (int(x + 100*vx), int(y + 100*vy))
                cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)
                cv2.circle(annotated, (int(x), int(y)), 5, (255, 0, 0), -1)
                
                # Draw line 2 in blue
                x, y, vx, vy = line2
                pt1 = (int(x - 100*vx), int(y - 100*vy))
                pt2 = (int(x + 100*vx), int(y + 100*vy))
                cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
                cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # Draw chosen combined line in green
                x, y, vx, vy = chosen_line
                pt1 = (int(x - 100*vx), int(y - 100*vy))
                pt2 = (int(x + 100*vx), int(y + 100*vy))
                cv2.line(annotated, pt1, pt2, (0, 255, 0), 3)  # Thicker green line
                cv2.circle(annotated, (int(x), int(y)), 7, (0, 255, 0), -1)  # Larger green circle
                
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "rgb8")
                self.detector_image_pub.publish(annotated_msg)
        else:
            # Fall back to single line detection
            line = self.detect_line(image)
            
            # Always publish a message
            msg = Line()
            if line is not None:
                msg.x, msg.y, msg.vx, msg.vy = line
            else:
                # No line detected
                msg.x, msg.y, msg.vx, msg.vy = -1.0, -1.0, 0.0, 1.0
            
            self.param_pub.publish(msg)
            
            # Visualization for single line
            if DISPLAY and line is not None:
                annotated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                x, y, vx, vy = line
                pt1 = (int(x - 100*vx), int(y - 100*vy))
                pt2 = (int(x + 100*vx), int(y + 100*vy))
                cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "rgb8")
                self.detector_image_pub.publish(annotated_msg)

    ##########
    # DETECT #
    ##########
    def detect_line(self, image):
        """ 
        Given an image, fit a line to biggest contour if it meets size requirements (otherwise return None)
        and return a parameterization of the line as a center point on the line and a vector
        pointing in the direction of the line.
            Args:
                - image = OpenCV image
            Returns: (x, y, vx, vy) where (x, y) is the centerpoint of the line in image and 
            (vx, vy) is a vector pointing in the direction of the line. Both values are given
            in downward camera pixel coordinates. Returns None if no line is found
        """
        
        # self.get_logger().info("trying to detect line...")

        h, w = image.shape
        
        kernel_size = 30
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 

        kernel_size = 20
        kernel2 = np.ones((kernel_size,kernel_size), np.uint8)

        # dilate + erode 
        image = cv2.dilate(image, kernel,iterations = 1)
        image = cv2.erode(image, kernel2,iterations = 1)

        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

        # threshold_msg = self.bridge.cv2_to_imgmsg(threshold, "mono8")
        # self.detector_image_pub.publish(threshold_msg)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_sort = lambda cnt: (max(cv2.minAreaRect(cnt)[1])) # sort by largest height/width 

        sorted_contours = sorted(contours, key=cnt_sort, reverse=True)

        if len(sorted_contours) > 0:
        
            all_points = np.vstack(sorted_contours[0])
            [vx, vy, x, y] = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
            return float(x), float(y), float(vx), float(vy)
        
    def detect_two_lines(self, image):
        """
        Detect two lines that are approximately at right angles.
        Returns parameters for both lines or None if not found.
        """
        h, w = image.shape[:2]
        
        # Your existing preprocessing...
        kernel_size = 40
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 
        kernel_size = 30
        kernel2 = np.ones((kernel_size,kernel_size), np.uint8)
        
        image = cv2.dilate(image, kernel, iterations = 1)
        image = cv2.erode(image, kernel2, iterations = 1)
        mask = cv2.inRange(image, LOW, HI)
        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)
        
        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            return None
        
        # Calculate angles for each line
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_params.append((angle, length, x1, y1, x2, y2))
        
        # Find two lines that are approximately perpendicular
        best_pair = None
        min_angle_diff = float('inf')
        
        for i in range(len(line_params)):
            for j in range(i+1, len(line_params)):
                angle1, len1, x1a, y1a, x2a, y2a = line_params[i]
                angle2, len2, x1b, y1b, x2b, y2b = line_params[j]
                
                # Calculate angle difference
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wraparound
                
                # Check if approximately perpendicular (90 degrees ± tolerance)
                if abs(angle_diff - np.pi/2) < np.pi/6:  # 30 degree tolerance
                    if abs(angle_diff - np.pi/2) < min_angle_diff:
                        min_angle_diff = abs(angle_diff - np.pi/2)
                        best_pair = (line_params[i], line_params[j])
        
        if best_pair is None:
            return None
        
        # Convert to your line format (x, y, vx, vy) for both lines
        line1, line2 = best_pair
        
        # For line 1
        _, _, x1a, y1a, x2a, y2a = line1
        vx1 = x2a - x1a
        vy1 = y2a - y1a
        length1 = np.sqrt(vx1**2 + vy1**2)
        vx1, vy1 = vx1/length1, vy1/length1  # Normalize
        x1, y1 = (x1a + x2a)/2, (y1a + y2a)/2  # Midpoint
        
        # For line 2
        _, _, x1b, y1b, x2b, y2b = line2
        vx2 = x2b - x1b
        vy2 = y2b - y1b
        length2 = np.sqrt(vx2**2 + vy2**2)
        vx2, vy2 = vx2/length2, vy2/length2  # Normalize
        x2, y2 = (x1b + x2b)/2, (y1b + y2b)/2  # Midpoint
        
        return {
            'line1': (float(x1), float(y1), float(vx1), float(vy1)),
            'line2': (float(x2), float(y2), float(vx2), float(vy2))
        }

def main(args=None):
    rclpy.init(args=args)
    detector = LineDetector()
    detector.get_logger().info("Line detector initialized")
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
