import cv2
import numpy as np
from tqdm import tqdm

from skeleton_tools.utils.constants import COLORS, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box


class Visualizer:
    def draw_json_skeletons(self, frame, skeletons, resolution, display_pid=True, display_bbox=True, is_normalized=False):
        width, height = resolution
        for i, s in enumerate(skeletons):
            pid = int(s['person_id']) if 'person_id' in s.keys() and not s['person_id'] == [-1] else i
            color = tuple(reversed(COLORS[pid % len(COLORS)]['value']))
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys() and s[src['name']]]:
                x = (np.array(s[src['name']][0::2]) * (width if is_normalized else 1)).astype(int)
                y = (np.array(s[src['name']][1::2]) * (height if is_normalized else 1)).astype(int)
                c = np.array(s[f'{src["name"]}_score'])
                pose = list(zip(x, y))
                self.draw_skeleton(frame, pose, c, src['layout'], color)

                if src['name'] == 'pose':
                    pose = np.array(pose).T
                    if display_pid:
                        x = pose[0][c > EPSILON]
                        y = pose[1][c > EPSILON]
                        x_center = x.mean() * 0.975
                        y_center = y.min() * 0.9
                        cv2.putText(frame, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                    if display_bbox:
                        bbox = bounding_box(pose, c)
                        bbox = (bbox[0]['min'], bbox[1]['min']), (bbox[0]['max'], bbox[1]['max'])
                        cv2.rectangle(frame, bbox[0], bbox[1], (255, 255, 255), thickness=1)

    def draw_skeleton(self, image, pose, score, skeleton_layout, color=None, join_emphasize=None, epsilon=0.05):
        if color is None:
            color = (0, 0, 255)
        for (v1, v2) in skeleton_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(image, tuple(pose[v1]), tuple(pose[v2]), color, thickness=2, lineType=cv2.LINE_AA)
        for i, (x, y) in enumerate(pose):
            if score[i] > epsilon:
                joint_size = join_emphasize[i] if join_emphasize else 2
                cv2.circle(image, (x, y), joint_size, (0, 60, 255), thickness=2)

    def make_video(self, video_path, skeleton, dst_file, delay=0):
        cap = cv2.VideoCapture(video_path)
        width, height, length, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dst_file, fourcc, fps, (width, height))

        curr_frame = 0
        with tqdm(total=length, ascii=True, desc="Writing video result") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if curr_frame >= delay:
                        self.draw_json_skeletons(frame, skeleton[curr_frame - delay]['skeleton'], (width, height), is_normalized=False)
                    out.write(frame)
                    curr_frame += 1
                    pbar.update(1)
                else:
                    break
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()