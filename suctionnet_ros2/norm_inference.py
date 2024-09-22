import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
import sys
from normal_std.normal_estimation import CameraInfo, create_point_cloud_from_depth_image, grid_sample
from normal_std.inference import estimate_suction, uniform_kernel

class NormStdInferencer:
    def __init__(self):
        # PrimeSense camera info
        self.camera_info = CameraInfo(640, 480, 525.8810348926615, 527.8853163315471, 321.0291284324178, 228.7422250324759, 1000)
    
    def infer(self, rgb_img : np.ndarray, depth_img : np.ndarray, seg_mask : np.ndarray):
        assert rgb_img.shape[:2] == depth_img.shape[:2]
        assert rgb_img.shape[:2] == seg_mask.shape[:2]
        assert len(depth_img.shape) == 2
        assert len(seg_mask.shape) == 2
        heatmap, normals, point_cloud = estimate_suction(depth_img, seg_mask, self.camera_info)

        # Visualize heatmap
        k_size = 15
        kernel = uniform_kernel(k_size)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        # print('kernel:', kernel.shape)
        heatmap = np.pad(heatmap, k_size//2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        # print('heatmap:', heatmap.shape)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=10)
        suction_directions = normals[idx0, idx1, :]
        suction_translations = point_cloud[idx0, idx1, :]
        self.visualize_heatmap(heatmap, rgb_img, idx0, idx1)
        self.visualize_suctions(suction_directions, suction_translations)

    def visualize_suctions(self, suction_directions, suction_translations):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Extract the components of the suction directions and translations
        trans_x = suction_translations[:, 0]
        trans_y = suction_translations[:, 1]
        trans_z = suction_translations[:, 2]

        dir_x = suction_directions[:, 0]
        dir_y = suction_directions[:, 1]
        dir_z = suction_directions[:, 2]

        # Create a 3D scatter plot for the translations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(trans_x, trans_y, trans_z, color='blue', label='Translations')

        # Use the quiver function to plot the directions as vectors
        ax.quiver(trans_x, trans_y, trans_z, dir_x, dir_y, dir_z, length=0.002, color='red', label='Directions')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Suction Directions and Translations')
        ax.legend()

        # Set equal aspect ratio
        max_range = np.array([trans_x.max()-trans_x.min(), trans_y.max()-trans_y.min(), trans_z.max()-trans_z.min()]).max() / 2.0
        mid_x = (trans_x.max()+trans_x.min()) * 0.5
        mid_y = (trans_y.max()+trans_y.min()) * 0.5
        mid_z = (trans_z.max()+trans_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()


    def visualize_heatmap(self, heatmap, rgb_img, idx0, idx1):
        import matplotlib.pyplot as plt
        score_image = heatmap
        score_image *= 255
        score_image = score_image.clip(0, 255)
        score_image = score_image.astype(np.uint8)
        score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
        rgb_image = 0.5 * rgb_img + 0.5 * score_image
        rgb_image = rgb_image.astype(np.uint8)

        # Plot dots on the idx0, idx1 positions
        for i in range(len(idx0)):
            cv2.circle(rgb_image, (idx1[i], idx0[i]), 5, (0, 255, 0), -1)
        
        # Display the image using Matplotlib
        plt.imshow(rgb_image)
        plt.title('Heatmap Visualization')
        plt.axis('off')  # Hide the axis
        plt.show()

if __name__ == "__main__":
    inferencer = NormStdInferencer()
    depth_dir = "/home/jinkai/Downloads/data/depth_image/first_scene/frame0000.jpg"
    rgb_dir = "/home/jinkai/Downloads/data/color_image/first_scene/frame0000.jpg"
    seg_dir = "/home/jinkai/Downloads/data/seg_masks/first_scene.png"
    rgb_img = cv2.imread(rgb_dir)
    depth_img = cv2.imread(depth_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 1000.0
    seg_mask = cv2.imread(seg_dir, cv2.IMREAD_GRAYSCALE).astype(bool)
    
    inferencer.infer(rgb_img, depth_img, seg_mask)