import os
import warnings

import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from skeleton_tools.utils.tools import read_pkl
from skeleton_tools.utils.skeleton_utils import get_boxes, get_iou

class YOLOTracker:
    def __init__(self, model='yolov8n.pt', grace_threshold=10):
        self.model = YOLO(model, verbose=False)
        self.grace_threshold = grace_threshold

    def track(self, video_path, reduce=True):
        results = self.model.track(video_path, verbose=False)
        if reduce:
            r = results[0]
            out = {
                'names': r.names,
                'orig_shape': r.orig_shape,
                'path': r.path
            }
            data = []
            for r in results:
                r = r.cpu()
                data.append({
                    'boxes': r.boxes,
                    'masks': r.masks,
                    'keypoints': r.keypoints,
                })
            out['data'] = data
            return out
        else:
            return results

    def match(self, skeleton, track_results):
        if type(skeleton) == str:
            skeleton = read_pkl(skeleton)
        kp = skeleton['keypoint']
        kps = skeleton['keypoint_score']
        M, T, _, _ = kp.shape
        assert np.abs(T - len(track_results)) < self.grace_threshold, f"Number of frames in skeleton ({T}) and track results ({len(track_results)}) do not match"
        S = np.min([T, len(track_results)])
        pids = np.ones((M, T)) * -1
        for i in range(S):
            boxes = get_boxes(kp[:, i, :, :], kps[:, i, :])
            detections = track_results[i].boxes
            idxs = [i for (i, c) in enumerate(detections.cls) if c == 0]
            detections = detections[idxs]
            if len(detections) == 0 or not detections.is_track:
                continue
            dboxes = detections.xywh
            dids = detections.id

            score_matrix = np.zeros((len(boxes), len(dboxes)))
            for _i, b in enumerate(boxes):
                for _j, db in enumerate(dboxes):
                    score_matrix[_i, _j] = get_iou(b, db)

            row_indices, col_indices = linear_sum_assignment(-score_matrix)  # Negate because we want to maximize
            for row, col in zip(row_indices, col_indices):
                pids[row, i] = dids[col]
        skeleton['person_ids'] = pids
        return skeleton

    def track_and_match(self, video_path, skeleton):
        results = self.track(video_path)
        skeleton = self.match(skeleton, results)
        return skeleton


if __name__ == '__main__':
    tracker = YOLOTracker()
    results = tracker.track(r'Z:\Users\TalBarami\models_outputs\samples\704767285_Cognitive_Control_300522_0848_1_5000_5500.mp4')
    skeleton = tracker.match(r'Z:\Users\TalBarami\models_outputs\samples\704767285_Cognitive_Control_300522_0848_1_5000_5500.pkl', results)
    print(1)
    # results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack