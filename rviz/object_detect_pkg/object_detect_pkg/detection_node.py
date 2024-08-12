import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge

class ImageSub(Node):
    def __init__(self):
        super().__init__("image_sub")
        self.sub = self.create_subscription(
            msg_type=Image,
            topic="/camera/depth/image_raw",
            callback=self.listener_callback,
            qos_profile=10
        )
        self.sub
        self.bridge = CvBridge()

    def listener_callback(self, data: Image):
        frame = self.bridge.imgmsg_to_cv2(data)
        cv.imshow("Camera Feed", frame)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_sub = ImageSub()
    rclpy.spin(image_sub)
    image_sub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()