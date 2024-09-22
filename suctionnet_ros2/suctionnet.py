from suctionnet_ros2.norm_inference import NormStdInferencer
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class SuctionNetNode(Node):
    def __init__(self):
        super().__init__("succionnet_node")
        self.norm_inferencer = NormStdInferencer()
        self.depth_image_topic = "/head_camera/depth/image_rect"
        self.rgb_image_topic = "/head_camera/rgb/image_rect_color"
        self.seg_mask_topic = "/segmentation_mask"
        self.graspPose_topic = "/grasp_pose_planned"
        self.depth_image = None
        self.rgb_image = None
        self.seg_mask = None
        self.create_service(PoseStamped, self.graspPose_topic, self.graspPose_callback)
        self.create_subscription(Image, self.depth_image_topic, self.depth_image_callback, 10)
        self.create_subscription(Image, self.rgb_image_topic, self.rgb_image_callback, 10)
        self.create_subscription(Image, self.seg_mask_topic, self.seg_mask_callback, 10)

def main():
    print('Hi from suctionnet_ros2.')


if __name__ == '__main__':
    main()
