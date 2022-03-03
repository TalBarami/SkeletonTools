import os
from os import path
from abc import ABC, abstractmethod

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skeleton_tools.utils.constants import COLORS, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_json


class BaseVisualizer(ABC):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, show_confidence=False):
        self.graph_layout = graph_layout
        self.display_pid = display_pid
        self.display_bbox = display_bbox
        self.denormalize = int(denormalize)
        self.decentralize = decentralize
        self.blur_face = blur_face
        self.show_confidence = show_confidence

    def draw_skeletons(self, frame, skeletons, scores, resolution, pids=None):
        img = np.copy(frame)

        if pids is None:
            pids = np.arange(skeletons.shape[0])

        if self.denormalize:
            skeletons *= resolution
        skeletons = skeletons.astype(int)

        if self.blur_face:
            for i, pose in enumerate(skeletons):
                face_joints = [0, 15, 16, 17, 18]
                pc = scores[i, face_joints]
                if (pc > 0).any():
                    ps = pose[:, face_joints][:, pc > 0].T
                    for p in ps:
                        img = self.blur_area(img, tuple(p), 100)

        for pose, score, pid in zip(skeletons, scores, pids):
            color = tuple(reversed(COLORS[(pid % len(COLORS))]['value']))

            img = self.draw_skeleton(img, pose, score, color)

            if self.display_pid:
                self.draw_pid(img, pose.T, score, pid, color)

            if self.display_bbox:
                self.draw_bbox(img, bounding_box(pose.T, score))
        return img

    def draw_skeleton(self, frame, pose, score, color=None, epsilon=0.05):
        img = np.copy(frame)
        if color is None:
            color = (0, 0, 255)
        for (v1, v2) in self.graph_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(img, tuple(pose[v1]), tuple(pose[v2]), color, thickness=2, lineType=cv2.LINE_AA)
        for i, (x, y) in enumerate(pose):
            if score[i] > epsilon:
                jcolor, jsize = (tuple(np.array(plt.cm.jet(score[i])) * 255), 4) if self.show_confidence else ((0, 60, 255), 2)
                cv2.circle(img, (x, y), jsize, jcolor, thickness=2)
        return img

    def draw_bbox(self, frame, bbox):
        center, r = bbox
        cv2.rectangle(frame, tuple((center - r).astype(int)), tuple((center + r).astype(int)), color=(255, 255, 255), thickness=1)

    def draw_pid(self, frame, pose, c, pid, color):
        if np.all(c < EPSILON):
            return
        x = pose[0][c > EPSILON]
        y = pose[1][c > EPSILON]
        x_center = x.mean() * 0.975
        y_center = y.min() * 0.9
        cv2.putText(frame, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

    def blur_area(self, frame, c, r):
        c_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(c_mask, c, r, 1, thickness=-1)
        mask = cv2.bitwise_and(frame, frame, mask=c_mask)
        img_mask = frame - mask
        blur = cv2.blur(frame, (50, 50))
        mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)  # mask
        final_img = img_mask + mask2
        return final_img

    def create_double_frame_video(self, video_path, skeleton_data, out_path, start=None, end=None):
        fps, frames_count, (width, height), kp, c, pids = self.get_video_info(video_path, skeleton_data)
        start = int(0 if start is None else start * fps)
        end = int(frames_count if end is None else end * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (2 * width, height))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in tqdm(range(start, end), desc="Writing video result"):
            ret, frame = cap.read()
            skel_frame = np.zeros_like(frame)
            if i < len(kp):
                skel_frame = self.draw_skeletons(skel_frame, kp[i], c[i], (width, height), pids[i])
            out.write(np.concatenate((frame, skel_frame), axis=1))
        cap.release()
        out.release()

    def create_video(self, video_path, skeleton_data, out_path):
        fps, length, (width, height), kp, c, pids = self.get_video_info(video_path, skeleton_data)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        cap = cv2.VideoCapture(video_path)

        for i in tqdm(range(length), desc="Writing video result"):
            ret, frame = cap.read()
            frame = self.draw_skeletons(frame, kp[i], c[i], (width, height), pids[i])
            out.write(frame)
        cap.release()
        out.release()

    @abstractmethod
    def get_video_info(self, video_path, skeleton_data):
        pass

    # def make_skeleton_video(self, skeleton, dst_file, display_pid=False, display_bbox=False, is_normalized=False, is_centralized=False, visualize_confidence=False):
    #     width, height = skeleton['resolution']
    #     fps = skeleton['fps']
    #     data = skeleton['data']
    #     length = len(data)
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(dst_file, fourcc, fps, (width, height))
    #     for i in tqdm(range(length), desc="Writing video result"):
    #         frame = np.zeros((width, height, 3), dtype='uint8')
    #         frame = self.draw_json_skeletons(frame, skeleton[i]['skeleton'], (width, height),
    #                                          display_pid=display_pid, display_bbox=display_bbox, is_normalized=is_normalized, is_centralized=is_centralized, visualize_confidence=visualize_confidence)
    #         out.write(frame)
    #     out.release()
    #
    # def make_video(self, video_path, skeleton, dst_file, delay=0, from_frame=None, to_frame=None, display_pid=False, display_bbox=False, is_normalized=False, is_centralized=False, blur_faces=False, visualize_confidence=False):
    #     cap = cv2.VideoCapture(video_path)
    #     width, height, length, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(dst_file, fourcc, fps, (2 * width, height))
    #     skeleton_data = skeleton['data']
    #     if from_frame is None:
    #         from_frame = 0
    #     if to_frame is None:
    #         to_frame = length
    #
    #     curr_frame = 0
    #     with tqdm(total=length, ascii=True, desc="Writing video result") as pbar:
    #         while cap.isOpened() and curr_frame < len(skeleton_data):
    #             ret, frame = cap.read()
    #             skel_frame = np.zeros_like(frame)
    #             if ret:
    #                 if curr_frame >= delay and curr_frame >= from_frame and curr_frame < to_frame:
    #                     skel_frame = self.draw_json_skeletons(skel_frame, skeleton_data[curr_frame - delay]['skeleton'], (width, height),
    #                                                           is_normalized=is_normalized, is_centralized=is_centralized, display_pid=display_pid, display_bbox=display_bbox, blur_face=blur_faces, visualize_confidence=visualize_confidence)
    #                     out.write(np.concatenate((frame, skel_frame), axis=1))
    #                 curr_frame += 1
    #                 pbar.update(1)
    #             else:
    #                 break
    #             if to_frame and curr_frame == to_frame:
    #                 break
    #     if cap is not None:
    #         cap.release()
    #     if out is not None:
    #         out.release()


if __name__ == '__main__':
    # org_root = r'D:\datasets\autism_center\skeletons\data'
    # skel_root = r'D:\datasets\autism_center\skeletons_filtered\data'
    # vids_root = r'D:\datasets\autism_center\segmented_videos'
    # out_dir = r'D:\datasets\autism_center\qa_vids'
    # bbox_root = r'C:\research\yolov5\runs\detect\100971247_Linguistic_210218_1109_3_Hand flapping_24965_25177\labels'
    # v = Visualizer()

    # file = '100971247_Linguistic_210218_1109_3_Hand flapping_24965_25177.json'
    # name, ext = path.splitext(file)
    # v_name = f'{name}.avi' if path.exists(path.join(vids_root, f'{name}.avi')) else f'{name}.mp4'
    # cap = cv2.VideoCapture(path.join(vids_root, v_name))
    # skel = read_json(path.join(org_root, file))['data']
    # for frame_info in skel:
    #     frame_info['skeleton'] = [{'person_id': s['person_id'], 'pose': s['pose'], 'pose_score': s['score']} for s in frame_info['skeleton']]
    # box_files = os.listdir(bbox_root)
    # i = 0
    # width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # d = ChildDetector()
    # box_json = d._collect_json(r'C:\research\yolov5\runs\detect\100971247_Linguistic_210218_1109_3_Hand flapping_24965_25177\labels')
    # cids = d._match_video(box_json, skel)
    #
    # while cap.isOpened() and i < len(skel):
    #     ret, frame = cap.read()
    #     if ret:
    #         cx, cy, w, h = box_json[i]['box']
    #         cx, cy, w, h = int(float(cx) * width), int(float(cy) * height), int(float(w) * 2 * width), int(float(h) * 2 * height)
    #         v.draw_bbox(frame, (np.array((cx, cy)), np.array((w, h))))
    #         # with open(path.join(bbox_root, box_files[i])) as f:
    #         #     boxs = [x.strip() for x in f.readlines()]
    #         # for line in boxs:
    #         #     l, cx, cy, w, h = line.split(' ')
    #         #     cx, cy, w, h = int(float(cx) * width), int(float(cy) * height), int(float(w) * 2 * width), int(float(h) * 2 * height)
    #         #     v.draw_bbox(frame, (np.array((cx, cy)), np.array((w, h))))
    #         #     cv2.putText(frame, l, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    #         frame = v.draw_json_skeletons(frame, skel[i]['skeleton'], (width, height), is_normalized=True)
    #         if cids[i] >= 0:
    #             try:
    #                 selected = [s for s in skel[i]['skeleton'] if s['person_id'] == cids[i]][0]
    #                 xx = np.array(selected['pose'][0::2]) * width
    #                 yy = np.array(selected['pose'][1::2]) * height
    #                 cc = np.array(selected['pose_score'])
    #                 cv2.putText(frame, "SELECTED", (int(xx[1]), int(yy[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    #             except IndexError:
    #                 print(1)
    #         cv2.imshow('', frame)
    #     i += 1
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    # cap.release()

    skel_root = r'D:\datasets\autism_center_take2\skeletons'
    vids_root = r'D:\datasets\autism_center_take2\segmented_videos'
    out_dir = r'D:\datasets\autism_center_take2\new_videos'
    v = BaseVisualizer()
    for file in os.listdir(skel_root):
        name, ext = path.splitext(file)
        v_name = f'{name}.avi' if path.exists(path.join(vids_root, f'{name}.avi')) else f'{name}.mp4'
        skel = read_json(path.join(skel_root, file))
        v.make_video(path.join(vids_root, v_name), skel, path.join(out_dir, v_name), is_normalized=False, display_pid=True)
