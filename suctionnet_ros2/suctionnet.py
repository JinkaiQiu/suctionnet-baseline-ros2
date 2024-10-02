from suctionnet_ros2.norm_inference import NormStdInferencer
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from perception_interfaces.srv import Sucpose
import numpy as np
from cv_bridge import CvBridge
import rclpy
from geometry_msgs.msg import Pose, Quaternion, Point

class SuctionNetNode(Node):
    def __init__(self):
        super().__init__("succionnet_node")
        self.norm_inferencer = NormStdInferencer()
        self.bridge = CvBridge()
        self.create_service(Sucpose, "sucpose_service", self.suctionnet_callback)
    
    def suctionnet_callback(self, request, response):
        rgb = request.color_image
        depth = request.depth_image
        seg_mask = request.segmask
        rgb_img = self.bridge.imgmsg_to_cv2(rgb, "rgb8")
        seg_mask = self.bridge.imgmsg_to_cv2(seg_mask, "mono8")
        depth_img = self.bridge.imgmsg_to_cv2(depth,"passthrough")
        depth_img = depth_img.astype(np.float32)
        depth_img /= 1000

        quat_values, trans_values = self.norm_inferencer.infer(rgb_img, depth_img, seg_mask)

        quat = Quaternion()
        quat.x = quat_values[0]
        quat.y = quat_values[1]
        quat.z = quat_values[2]
        quat.w = quat_values[3]

        trans = Point()
        trans.x = trans_values[0]
        trans.y = trans_values[1]
        trans.z = trans_values[2]

        res = Pose()
        res.orientation = quat
        res.position = trans
 
        response.pose = res
        return response

def main():
    rclpy.init()
    node = SuctionNetNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
