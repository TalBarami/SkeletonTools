import cv2
import numpy as np
from tqdm import tqdm

from skeleton_tools.skeleton_visualization.base_visualizer import BaseVisualizer


class MMPoseVisualizer(BaseVisualizer):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, show_confidence=False):
        super().__init__(graph_layout, display_pid, display_bbox, denormalize, decentralize, blur_face, show_confidence)

    def get_video_info(self, video_path, skeleton_data):
        fps = skeleton_data['fps']
        length = skeleton_data['total_frames']
        width, height = skeleton_data['original_shape']
        width, height = int(width), int(height)
        kp = skeleton_data['keypoint'].transpose((1, 0, 2, 3))
        c = skeleton_data['keypoint_score'].transpose((1, 0, 2))
        child_ids = skeleton_data['child_ids'] if 'child_ids' in skeleton_data.keys() else None
        pids = [np.arange(p.shape[0]) for p in kp]

        return fps, length, (width, height), kp, c, pids, child_ids