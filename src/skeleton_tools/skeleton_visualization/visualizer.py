import cv2
import numpy as np
from tqdm import tqdm

from skeleton_tools.utils.constants import COLORS, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box


class Visualizer:
    def draw_json_skeletons(self, frame, skeletons, resolution, display_pid=True, display_bbox=True, is_normalized=False, pid_colors=True, blur_face=False):
        width, height = resolution
        img = np.copy(frame)
        if blur_face:
            for i, s in enumerate(skeletons):
                x = (np.array(s['pose'][0::2]) * (width if is_normalized else 1)).astype(int)
                y = (np.array(s['pose'][1::2]) * (height if is_normalized else 1)).astype(int)
                c = np.array(s['pose_score'])
                pose = np.array(list(zip(x, y))).T
                l = [0, 15, 16, 17, 18]
                pc = c[l]
                if (pc > 0).any():
                    ps = pose[:, l][:, pc > 0].T
                    for p in ps:
                        img = self.blur_area(img, tuple(p), 100)

                # [0, 1, 15, 16, 17, 18]
                # if c[0] > EPSILON:
                #     center = (pose[0][0], pose[1][0])
                #     img = self.blur_area(img, center, 100)
                # else:
                #     l = [0, 1, 15, 16, 17, 18]
                #     pc = c[l]
                #     if (pc > 0).any():
                #         px = pose[0][l][pc > 0]
                #         py = pose[1][l][pc > 0]
                #         center = (np.mean(px).astype(int), np.mean(py).astype(int))
                #         img = self.blur_area(img, center, 150)

        for i, s in enumerate(skeletons):
            pid = int(s['person_id']) if 'person_id' in s.keys() and not s['person_id'] == [-1] else i
            color = tuple(reversed(COLORS[(pid % len(COLORS)) if pid_colors else 0]['value']))
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys() and s[src['name']]]:
                x = (np.array(s[src['name']][0::2]) * (width if is_normalized else 1)).astype(int)
                y = (np.array(s[src['name']][1::2]) * (height if is_normalized else 1)).astype(int)
                c = np.array(s[f'{src["name"]}_score'])
                pose = list(zip(x, y))
                img = self.draw_skeleton(img, pose, c, src['layout'], color)

                if src['name'] == 'pose':
                    pose = np.array(pose).T
                    if display_pid:
                        x = pose[0][c > EPSILON]
                        y = pose[1][c > EPSILON]
                        x_center = x.mean() * 0.975
                        y_center = y.min() * 0.9
                        cv2.putText(img, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                    if display_bbox:
                        self.draw_bbox(frame, bounding_box(pose, c))
                        # bbox = bounding_box(pose, c)
                        # bbox = (bbox[0]['min'], bbox[1]['min']), (bbox[0]['max'], bbox[1]['max'])
                        # cv2.rectangle(frame, bbox[0], bbox[1], (255, 255, 255), thickness=1)

    def draw_bbox(self, frame, bbox):
        center, r = bbox
        cv2.rectangle(frame, tuple((center - r).astype(int)), tuple((center + r).astype(int)), color=(255, 255, 255), thickness=1)

    def blur_area(self, frame, c, r):
        c_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(c_mask, c, r, 1, thickness=-1)
        mask = cv2.bitwise_and(frame, frame, mask=c_mask)
        img_mask = frame - mask
        blur = cv2.blur(frame, (50, 50))
        mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)  # mask
        final_img = img_mask + mask2
        return final_img

    # def blur_faces(self, frame):
    #     face_detect = cv2.CascadeClassifier(r'resources/haarcascade_frontalface_default.xml')
    #     face_data = face_detect.detectMultiScale(frame, 1.3, 5)
    #     for (x, y, w, h) in face_data:
    #         print('detected')
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         roi = frame[y:y + h, x:x + w]
    #         # applying a gaussian blur over this new rectangle area
    #         roi = cv2.GaussianBlur(roi, (23, 23), 30)
    #         # impose this blurred image on original image to get final image
    #         frame[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
    #     return frame

    def draw_skeleton(self, frame, pose, score, skeleton_layout, color=None, join_emphasize=None, epsilon=0.05):
        img = np.copy(frame)
        if color is None:
            color = (0, 0, 255)
        for (v1, v2) in skeleton_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(img, tuple(pose[v1]), tuple(pose[v2]), color, thickness=2, lineType=cv2.LINE_AA)
        for i, (x, y) in enumerate(pose):
            if score[i] > epsilon:
                joint_size = join_emphasize[i] if join_emphasize else 2
                cv2.circle(img, (x, y), joint_size, (0, 60, 255), thickness=2)
        return img

    def make_video(self, video_path, skeleton, dst_file, delay=0, from_frame=None, to_frame=None, display_pid=False, display_bbox=False, is_normalized=False, pid_colors=True, blur_faces=False):
        cap = cv2.VideoCapture(video_path)
        width, height, length, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dst_file, fourcc, fps, (width, height))
        if from_frame is None:
            from_frame = 0
        if to_frame is None:
            to_frame = length

        curr_frame = 0
        with tqdm(total=length, ascii=True, desc="Writing video result") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if curr_frame >= delay and curr_frame >= from_frame and curr_frame < to_frame:
                        frame = self.draw_json_skeletons(frame, skeleton[curr_frame - delay]['skeleton'], (width, height), is_normalized=is_normalized, display_pid=display_pid, display_bbox=display_bbox, pid_colors=pid_colors,
                                                 blur_face=blur_faces)
                        out.write(frame)
                    curr_frame += 1
                    pbar.update(1)
                else:
                    break
                if to_frame and curr_frame == to_frame:
                    break
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
