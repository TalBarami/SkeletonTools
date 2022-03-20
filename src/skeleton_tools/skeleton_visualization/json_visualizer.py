import cv2
import numpy as np
from skeleton_tools.skeleton_visualization.base_visualizer import BaseVisualizer
from skeleton_tools.utils.constants import COLORS, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import get_video_properties


class JsonVisualizer(BaseVisualizer):
    def get_video_info(self, video_path, skeleton_json):
        if video_path is None:
            (width, height), fps, length = skeleton_json['resolution'], skeleton_json['fps'], skeleton_json['length']
        else:
            (width, height), fps, length = get_video_properties(video_path)

        kp = []
        c = []
        pids = []
        for frame_info in skeleton_json['data']:
            skeletons = frame_info['skeleton']
            kp.append(np.array([list(zip(s['pose'][::2], s['pose'][1::2])) for s in skeletons]))
            c.append(np.array([s['pose_score'] for s in skeletons]))
            pids.append([s['person_id'] for s in skeletons])

        return fps, length, (width, height), kp, c, pids