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
LOW = None  # Lower image thresholding bound
HI = None   # Upper image thresholding bound
LENGTH_THRESH = None  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True

class LineDetector(Node):
    def __init__(self):
        super().__init__('detector')

        # A subscriber to the topic '/aero_downward_camera/image'
        self.camera_sub = self.create_subscription(
            Image,
            '/camera_1/image_raw',
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
            Args: 
                - msg = ROS Image message
        """

        # self.get_logger().info("camera sub callback function!!")

        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Detect line in the image. detect returns a parameterize the line (if one exists)
        line = self.detect_line(image)

        # If a line was detected, publish the parameterization to the topic '/line/param'
        if line is not None:
            msg = Line()
            msg.x, msg.y, msg.vx, msg.vy = line
            # Publish param msg
            self.param_pub.publish(msg)
            # self.get_logger().info("param pub !!")

        # Publish annotated image if DISPLAY is True and a line was detected
        if DISPLAY and line is not None:
            annotated = image
            x, y, vx, vy = line
            pt1 = (int(x - 100*vx), int(y - 100*vy))
            pt2 = (int(x + 100*vx), int(y + 100*vy))
            # pt1 = int(x), int(y)
            # pt2 = int(vx), int(vy)
            cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Convert to ROS Image message and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
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

        h, w, _ = image.shape
        
        kernel_size = 40
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 

        kernel_size = 30
        kernel2 = np.ones((kernel_size,kernel_size), np.uint8)

        LOW = np.array([250, 250, 250])  # Lower image thresholding bound
        HI = np.array([255, 255, 255])   # Upper image thresholding bound

        # dilate + erode 
        image = cv2.dilate(image, kernel,iterations = 1)
        image = cv2.erode(image, kernel2,iterations = 1)

        # Apply white mask to filter out external colors
        mask = cv2.inRange(image, LOW, HI)
        image = cv2.bitwise_and(image, image, mask=mask)

        image = cv2.GaussianBlur(image, (5,5), 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)

        # threshold_msg = self.bridge.cv2_to_imgmsg(threshold, "mono8")
        # self.detector_image_pub.publish(threshold_msg)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_sort = lambda cnt: (max(cv2.minAreaRect(cnt)[1])) # sort by largest height/width 

        sorted_contours = sorted(contours, key=cnt_sort, reverse=True)

        if len(sorted_contours) > 0:
        
            all_points = np.vstack(sorted_contours[0])
            [vx, vy, x, y] = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
            return float(x), float(y), float(vx), float(vy)
    
    def find_inliers(self, m, b, shape):
        height, width = shape

        # x = 0 (left edge), compute y
        x1 = 0
        y1 = m * x1 + b

        # x = width - 1 (right edge), compute y
        x2 = width
        y2 = m * x2 + b

        # Clip y values to stay within the image bounds
        y1 = max(0, min(height - 1, int(round(y1))))
        y2 = max(0, min(height - 1, int(round(y2))))

        return (x1, x2, y1, y2)

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
