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
from normal_std.inference import estimate_suction
from .util import CameraInfo
from .util import SuctionNetUtils as SNU

class NormStdInferencer:
    def __init__(self):
        # PrimeSense camera info
        self.camera_info = CameraInfo(640, 480, 525.8810348926615, 527.8853163315471, 321.0291284324178, 228.7422250324759, 1000)
    
    def infer(self, rgb_img : np.ndarray, depth_img : np.ndarray, seg_mask=None):
        assert rgb_img.shape[:2] == depth_img.shape[:2]
        assert rgb_img.shape[:2] == seg_mask.shape[:2]
        assert len(depth_img.shape) == 2
        assert len(seg_mask.shape) == 2
        heatmap, normals, point_cloud = estimate_suction(depth_img, seg_mask, self.camera_info)

        # Visualize heatmap
        k_size = 15
        kernel = SNU.uniform_kernel(k_size)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        # print('kernel:', kernel.shape)
        heatmap = np.pad(heatmap, k_size//2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        # print('heatmap:', heatmap.shape)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        suction_scores, idx0, idx1 = SNU.grid_sample(heatmap, down_rate=10, topk=10)
        
        if seg_mask is not None:
            suctions, idx0, idx1 = SNU.filter_suctions(suction_scores, idx0, idx1, seg_mask)

        suction_directions = normals[idx0, idx1, :]
        suction_translations = point_cloud[idx0, idx1, :]
        # visualize_heatmap(heatmap, rgb_img, idx0, idx1)
        # visualize_suctions(suction_directions, suction_translations)
        
        # average suction direction and translation after filtering outliers
        suction_direction = np.mean(SNU.remove_outliers(suction_directions, 1), axis=0)
        suction_translation = np.mean(SNU.remove_outliers(suction_translations, 1), axis=0)
        suction_quat = SNU.unit_vect_to_quat(suction_direction)

        return suction_quat, suction_translation 

if __name__ == "__main__":
    inferencer = NormStdInferencer()
    depth_dir = "/home/jinkai/Downloads/data/depth_image/first_scene/frame0000.jpg"
    rgb_dir = "/home/jinkai/Downloads/data/color_image/first_scene/frame0000.jpg"
    seg_dir = "/home/jinkai/Downloads/data/seg_masks/first_scene.png"
    rgb_img = cv2.imread(rgb_dir)
    depth_img = cv2.imread(depth_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 1000.0
    seg_mask = cv2.imread(seg_dir, cv2.IMREAD_GRAYSCALE).astype(bool)
    
    inferencer.infer(rgb_img, depth_img, seg_mask)