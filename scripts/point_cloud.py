#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2 as cv
from cv_bridge import CvBridge
# from yolov3_opencv import util
import tf2_ros
import numpy as np

class PoinCloud(Node):
    def __init__(self):
        super().__init__("point_cloud")
        self.sub = self.create_subscription(
            msg_type=PointCloud2,
            topic="/camera/points",
            callback=self.image_callback,
            qos_profile=10
        )

        self.bridge = CvBridge()

    def image_callback(self, data: PointCloud2):
        #convert point cloud2 to numpy array
        np; pointcloud = np.frombuffer(data.data, dtype=np.float32).reshape(-1, 4)

        # Extract rgb and depth information
        rgb_data = pointcloud[:, :3].astype(np.uint8)
        depth_data = pointcloud[:, 3]
        
        # create RGB and depth images
        rgb_image = cv.cvtColor(rgb_data.reshape(data.height, data.width, 3), cv.COLOR_RGB2BGR)
        depth_image = cv.normalize(depth_data.reshape(data.height, data.width), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UCI)

        cv.imshow('Image', rgb_image)
        cv.imshow("depth", depth_image)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_sub = PoinCloud()
    rclpy.spin(image_sub)
    image_sub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()