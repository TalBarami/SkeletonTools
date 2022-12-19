import cv2
import numpy as np
from tqdm import tqdm

from skeleton_tools.skeleton_visualization.skeleton_visualizer import SkeletonVisualizer


class MMPoseVisualizer(SkeletonVisualizer):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, show_confidence=False):
        super().__init__(graph_layout, display_pid, display_bbox, denormalize, decentralize, blur_face, show_confidence)

    def prepare(self, video_path, skeleton_data):
        skeleton_data['length'] = skeleton_data['total_frames']
        width, height = skeleton_data['original_shape']
        skeleton_data['width'] = int(width)
        skeleton_data['height'] = int(height)
        skeleton_data['keypoint'] = skeleton_data['keypoint'].transpose((1, 0, 2, 3))
        skeleton_data['keypoint_score'] = skeleton_data['keypoint_score'].transpose((1, 0, 2))
        skeleton_data['pids'] = [np.arange(p.shape[0]) for p in skeleton_data['keypoint']]

        return skeleton_data