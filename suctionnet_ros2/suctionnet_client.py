import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np

# Assume the service type is suc_pose_interface.srv.SucPose
from perception_interfaces.srv import Sucpose

class SucPoseClient(Node):
    def __init__(self):
        super().__init__('suc_pose_client')
        self.cli = self.create_client(Sucpose, '/sucpose_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = Sucpose.Request()
        self.bridge = CvBridge()

    def send_request(self):
        # Load images from local directory
        depth_dir = "/home/jinkai/Downloads/data/depth_image/first_scene/frame0000.jpg"
        rgb_dir = "/home/jinkai/Downloads/data/color_image/first_scene/frame0000.jpg"
        seg_dir = "/home/jinkai/Downloads/data/seg_masks/first_scene.png"

        # Read images using OpenCV
        color_image_cv = cv2.imread(rgb_dir, cv2.IMREAD_COLOR)
        depth_image_cv = cv2.imread(depth_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 1000.0
        segmask_cv = cv2.imread(seg_dir, cv2.IMREAD_GRAYSCALE).astype(bool)
        segmask_cv = segmask_cv.astype(np.uint8) * 255

        # Convert BGR to RGB for color image
        color_image_cv = cv2.cvtColor(color_image_cv, cv2.COLOR_BGR2RGB)

        # Convert depth image to float32
        depth_image_cv = depth_image_cv.astype('float32')

        # Convert OpenCV images to ROS Image messages
        color_image_msg = self.bridge.cv2_to_imgmsg(color_image_cv, encoding="rgb8")
        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image_cv, encoding="32FC1")
        segmask_msg = self.bridge.cv2_to_imgmsg(segmask_cv, encoding="mono8")

        # Set the request fields
        self.req.color_image = color_image_msg
        self.req.depth_image = depth_image_msg
        self.req.segmask = segmask_msg

        # Prepare CameraInfo message
        camera_info = CameraInfo()
        self.req.camera_info = camera_info

        # Call the service
        self.future = self.cli.call_async(self.req)

        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            self.get_logger().info(f'Received response: {self.future.result()}')
        else:
            self.get_logger().error('Exception while calling service: %r' % self.future.exception())

def main(args=None):
    rclpy.init(args=args)
    client = SucPoseClient()
    client.send_request()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
