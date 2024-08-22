#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge
# from yolov3_opencv import util
import tf2_ros
import numpy as np

class ImageSub(Node):
    def __init__(self):
        super().__init__("image_sub")
        self.sub = self.create_subscription(
            msg_type=Image,
            topic="/camera/image_raw",
            callback=self.image_callback,
            qos_profile=10
        )
        # self.depth_sub = self.create_subscription(
        #     msg_type=Image,
        #     topic="/camera/depth/image_raw",
        #     callback=self.depth_callback,
        #     qos_profile=10
        # )
        self.sub
        # self.depth_sub
        self.bridge = CvBridge()
        self.min_dist = "None"

        # load weigths, cfg file and class names
        package_dir = get_package_share_directory("intel_sys")
        self.model_cfg_path = os.path.join(package_dir, "config", "yolov3-tiny.cfg")
        self.model_weights_path = os.path.join(package_dir, "config", "yolov3-tiny.weights")
        self.class_names_path = os.path.join(package_dir, "config", "coco.names")

        with open(self.class_names_path, 'r') as f:
            self.class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
            f.close()

        # load model
        self.net = cv.dnn.readNetFromDarknet(self.model_cfg_path, self.model_weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.depth_image = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def image_callback(self, data: Image):
        np; frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        H, W, _ = frame.shape
        blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # change output shape with respecy your model
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.class_names[class_ids[i]])
                self.x, self.y, self.w, self.h = x, y, w, h
                self.depth_sub = self.create_subscription(
                    msg_type=Image,
                    topic="/camera/depth/image_raw",
                    callback=self.depth_callback,
                    qos_profile=1
                )
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, label + self.min_dist, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv.imshow('Image', frame)
        cv.waitKey(1)
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        img = self.depth_image[self.x:self.x + self.w, self.y:self.y + self.h]
        # cv.imshow("depth",img)
        # cv.waitKey(1)
        self.min_dist = str(np.nanmin(self.depth_image))



def main(args=None):
    rclpy.init(args=args)
    image_sub = ImageSub()
    rclpy.spin(image_sub)
    image_sub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()