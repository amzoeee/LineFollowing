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
            Args: 
                - msg = ROS Image message
        """

        # self.get_logger().info("camera sub callback function!!")

        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "mono8")

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
            # Draw the detected line on a color version of the image
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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

        '''
        TODO: Implement computer vision to detect a line (look back at last week's labs)
        TODO: Retrieve x, y pixel coordinates and vx, vy collinear vector from the detected line (look at cv2.fitLine)
        TODO: Populate the Line custom message and publish it to the topic '/line/param'
        '''
        # self.get_logger().info("trying to detect line...")

        h, w = image.shape
        
        kernelsize = 6 # because i'm rlly lazy
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations=4)

        _, threshold = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY)

        threshold_msg = self.bridge.cv2_to_imgmsg(threshold, "mono8")
        # self.detector_image_pub.publish(threshold_msg)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        # self.get_logger().info("number of contours: " + str(len(contours)))
        height, width = threshold.shape[:2]
        contour_only_image = np.zeros((height, width, 3), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # if area > 8500 or (h > 80 and w < 70):
            if True:
                # self.get_logger().info("contour area: " + str(area))
                valid_contours.append(cnt)

        if len(valid_contours) > 0:
            valid_contours = sorted(valid_contours, key=lambda cnt: cv2.contourArea(cnt))
            # # choose biggest contour by area TODO: change probably
            # cv2.drawContours(contour_only_image, [valid_contours[0]], 0, (255,255,255), -1)

            # gray_image = cv2.cvtColor(contour_only_image, cv2.COLOR_BGR2GRAY)

            # # image = self.bridge.cv2_to_imgmsg(gray_image, "mono8")
            # # self.detector_image_pub.publish(image)

            # points = np.argwhere(gray_image)

            # m, b = self.calculate_regression(points)

            # # self.get_logger().info(str(m))
            
            # x1, y1, x2, y2 = self.find_inliers(m,b, gray_image.shape)
            
            # # # reflect around center point?? idk if coordinate grr
            # # x1 = w-x1
            # # x2 = w-x2
            # # y1 = h-y1
            # # y2 = h-y2

            # # # find midpoint of the line
            # x = (x1 + x2)/2 if x1 > x2 else (x2 + x1)/2
            # y = (y1 + y2)/2 if y1 > y2 else (y2 + y1)/2

            # # x = x1
            # # y = y1

            # # TODO: maybe needs negating or something else idk coordinate systems
            # vx = 1/np.sqrt(1 + m**2)
            # vy = m/np.sqrt(1 + m**2)

            [vx, vy, x, y] = cv2.fitLine(valid_contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
            
            return float(x), float(y), float(vx), float(vy)
            # return x1, y1, x2, y2

    def calculate_regression(self, points): # input is the result of np.argwhere
        # convert points to float
        points = points.astype(float) #TODO (see astype, https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html)
        
        xs = points[:, 1]
        ys = points[:, 0]
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)

        xy_mean = np.mean(xs * ys)

        x_squared_mean = np.mean(np.square(xs))

        m = (x_mean*y_mean - xy_mean) / (np.square(x_mean) - x_squared_mean)
        
        b = y_mean - m*x_mean

        return (m,b)  
    
    def find_inliers(self, m, b, shape):
        width = shape[1]
        height = shape[0]

        if b < 0: # if right side would go below 
            y1 = 0
            x1 = (y1-b)/m

            # the other point: either left side or above 
            if m*width + b > height: # if left side would go above
                y2 = height
                x2 = (y2-b)/m

            else: # left side would be fine
                x2 = width
                y2 = m*width + b

        elif shape[1] < b: # if right side would go above
            y1 = height
            x1 = (y1-b)/m

            # the other point: either left side or below 
            if m*width + b < 0: # if left side would go below
                y2 = 0
                x2 = (y2-b)/m

            else: # left side would be fine
                x2 = width
                y2 = m*width + b

        else: # if right side is fine (0 <= b <= shape[1])
            x1, y1 = 0, b

            # left side has no limitations
            if (m*width + b) < 0: # if it would go below
                y2 = 0
                x2 = (y2-b)/m

            elif shape[1] < (m*width + b): # if it would go above
                y2 = height
                x2 = (y2-b)/m

            else: # it'll go through the left edge (0 < m*max_b + b < shape[1])
                x2 = width
                y2 = m*width + b
            
        return x1, y1, x2, y2

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