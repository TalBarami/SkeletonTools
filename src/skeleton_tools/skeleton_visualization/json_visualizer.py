import cv2
import numpy as np
from skeleton_tools.skeleton_visualization.base_visualizer import BaseVisualizer
from skeleton_tools.utils.constants import COLORS, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import get_video_properties


class JsonVisualizer(BaseVisualizer):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, show_confidence=False):
        super().__init__(graph_layout, display_pid, display_bbox, denormalize, decentralize, blur_face, show_confidence)

    def get_video_info(self, video_path, skeleton_data):
        (width, height), fps, length = get_video_properties(video_path)

        kp = []
        c = []
        pids = []
        for frame_info in skeleton_data['data']:
            skeletons = frame_info['skeleton']
            kp.append(np.array([list(zip(s['pose'][::2], s['pose'][1::2])) for s in skeletons]))
            c.append(np.array([s['pose_score'] for s in skeletons]))
            pids.append([s['person_id'] for s in skeletons])
        # kp = np.array(kp)
        # c = np.array(c)
        # pids = np.array(pids)
        # kp = skeleton_data['keypoint'].transpose((1, 0, 3, 2))
        # c = skeleton_data['keypoint_score'].transpose((1, 0, 2))
        # pids = np.arange(kp.shape[1])

        return fps, length, (width, height), kp, c, pids