import sys
from argparse import Namespace

import numpy as np
from skeleton_tools.utils.skeleton_utils import bounding_box, get_iou
from skeleton_tools.utils.tools import read_pkl, write_pkl
from torch.utils.data import DataLoader

from skeleton_tools.datasets.iterable_video_dataset import IterableVideoDataset
from tqdm import tqdm

sys.path.append(r'D:\repos\MiVOLO')
from mivolo.predictor import Predictor

def get_boxes(kp, score, method='xywh'):
    M = kp.shape[0]
    return [bounding_box(kp[i].T, score[i], method=method).reshape(-1) for i in range(M)]


def find_nearest(cb, boxes):
    iou = [get_iou(cb, b) for b in boxes]
    nearest = np.argmax(iou)
    return nearest, np.max(iou)


class AgeDetector:
    def __init__(self, batch_size=128, device='cuda:0', detector_weights=r"D:\repos\MiVOLO\models\yolov8x_person_face.pt", checkpoint=r"D:\repos\MiVOLO\models\model_imdb_cross_person_4.22_99.46.pth.tar"):
        args = {
            'detector_weights': detector_weights,
            'checkpoint': checkpoint,
            'device': device,
            'with_persons': True,
            'disable_faces': False,
            'draw': False
        }
        self.predictor = Predictor(Namespace(**args), verbose=False)
        self.batch_size = batch_size

        self.tolerance = 100
        self.age_threshold = 18
        self.iou_threshold = 0.1
        self.grace_distance = 30

    def detect(self, video_path):
        dataset = IterableVideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        result = []
        i = 0
        for frames_batch in dataloader:
            for im in frames_batch:
                detections, _ = self.predictor.recognize(im)
                n = detections.n_objects
                result.append({
                    'n_objects': n,
                    'n_faces': detections.n_faces,
                    'n_persons': detections.n_persons,
                    'ages': detections.ages,
                    'genders': detections.genders,
                    'gender_scores': detections.gender_scores,
                    'bboxes': [detections.get_bbox_by_ind(j).detach().cpu().numpy() for j in range(n)],
                    'face_to_person_map': detections.face_to_person_map,
                })
                i += 1
        return result

    def match_skeleton(self, skeleton, detections):
        kp = skeleton['keypoint']
        kps = skeleton['keypoint_score']

        _, T, _, _ = kp.shape
        adj = len(detections) - T
        if np.abs(adj) > self.tolerance:
            raise IndexError(f'Length mismatch: skeleton({T}) - video({len(detections)})')
        # if adj <= 0:
        #     detections = detections + [detections[-1]] * np.abs(adj)
        # else:
        #     detections = detections[adj:]

        skeleton['child_ids'] = np.ones(T) * -1
        skeleton['child_detected'] = np.zeros(T)
        skeleton['child_bbox'] = np.zeros((T, 4))

        cids = skeleton['child_ids']
        detected = skeleton['child_detected']
        boxes = skeleton['child_bbox']
        # _, detections = list(zip(*detections))
        self._straight_match(detections, kp, kps, cids, detected, boxes)
        self._interpolate(detections, kp, kps, cids, detected, boxes)
        return skeleton

    def _straight_match(self, detections, kp, kps, cids, detected, boxes):
        for i, det in tqdm(enumerate(detections), desc='Skeleton Matcher', total=len(detections)):
            n = det['n_objects']
            if n == 0:
                continue
            else:
                argmin_age = np.argmin(det['ages'])
                min_age = det['ages'][argmin_age]
                if min_age < self.age_threshold:
                    child_box = det['bboxes'][argmin_age]
                else:
                    continue
            boxes[i] = child_box
            cid, iou = find_nearest(child_box, get_boxes(kp[:, i, :, :], kps[:, i, :], method='xyxy'))
            if iou < self.iou_threshold:
                continue
            detected[i] = min_age
            cids[i] = cid


    def _interpolate(self, detections, kp, kps, cids, detected, boxes):
        env = [{} for _ in detected]

        def scan(lst, key, reverse):
            tmp = None
            for i, age in reversed(list(enumerate(lst))) if reverse else enumerate(lst):
                if age != 0 and age < self.age_threshold:
                    tmp = i
                env[i][key] = tmp

        scan(detected, 'prev', reverse=False)
        scan(detected, 'next', reverse=True)

        for i, det in tqdm(enumerate(detections), desc='Interpolate', total=len(detections)):
            if detected[i] != 0 and detected[i] < self.age_threshold:
                continue
            prev, next = env[i]['prev'], env[i]['next']
            if not ((prev and np.abs(prev - i) < self.grace_distance) or (next and np.abs(next - i) < self.grace_distance)):
                continue
            j = prev if next is None else next if prev is None else prev if abs(i - prev) >= abs(next - i) else next
            _det = detections[j]
            argmin_age = np.argmin(_det['ages'])
            min_age = _det['ages'][argmin_age]
            child_box = _det['bboxes'][argmin_age]
            cid, iou = find_nearest(child_box, get_boxes(kp[:, i, :, :], kps[:, i, :], method='xyxy'))
            if iou < self.iou_threshold:
                continue
            # adults = df[df['class'] == 0]
            # adults_matches = [(idx, find_nearest(adult_box, get_boxes(kp[:, i, :, :], kps[:, i, :]))) for idx, adult_box in adults.iterrows()]
            # conflicts = [(idx, a, iou) for idx, (a, iou) in adults_matches if a == candidate and iou > self.iou_threshold]
            # if any(rival_iou > candidate_iou and \
            #        not get_iou(get_box(child_box), get_box(adults.loc[idx])) > self.similarity_threshold for idx, _, rival_iou in conflicts):
            #     continue
            cids[i] = cid
            boxes[i] = child_box