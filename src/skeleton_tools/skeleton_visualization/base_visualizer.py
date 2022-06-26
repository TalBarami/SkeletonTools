import os
from os import path
from abc import ABC, abstractmethod

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
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

    def draw_skeletons(self, frame, skeletons, scores, resolution=None, pids=None):
        img = np.copy(frame)

        if pids is None:
            pids = np.arange(skeletons.shape[0])

        if self.denormalize:
            skeletons *= resolution
        skeletons = skeletons.astype(int)

        if self.blur_face:
            for i, (pose, score) in enumerate(zip(skeletons, scores)):
                face_joints = [k for k, v in self.graph_layout.joints().items() if any([s in v for s in ['Eye', 'Ear', 'Nose']])]
                pc = score[face_joints]
                if (pc > 0).any():
                    ps = pose[face_joints][pc > 0]
                    for p in ps:
                        img = self.blur_area(img, tuple(p), 100)

        for pose, score, pid in zip(skeletons, scores, pids):
            color = tuple(reversed(COLORS[(pid % len(COLORS))]['value']))
            if img.shape[-1] > 3:
                color += (255,)

            img = self.draw_skeleton(img, pose, score, color)

            if self.display_pid:
                self.draw_pid(img, pose.T, score, pid, color)

            if self.display_bbox:
                self.draw_bbox(img, bounding_box(pose.T, score))
        return img

    def draw_skeleton(self, frame, pose, score, edge_color=None, epsilon=0.05):
        img = np.copy(frame)
        if edge_color is None:
            edge_color = (0, 0, 255)
        joint_color = (0, 60, 255)
        if img.shape[-1] > 3:
            joint_color += (255,)
        for (v1, v2) in self.graph_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(img, tuple(pose[v1]), tuple(pose[v2]), edge_color, thickness=2, lineType=cv2.LINE_AA)
        # for i, (x, y) in enumerate(pose):
        #     if score[i] > epsilon:
        #         jcolor, jsize = (tuple(np.array(plt.cm.jet(score[i])) * 255), 4) if self.show_confidence else (joint_color, 2)
        #         cv2.circle(img, (x, y), jsize, jcolor, thickness=2)
        return img

    def draw_bbox(self, frame, bbox):
        center, r = bbox
        bcolor = (255, 255, 255)
        if frame.shape[-3] > 3:
            bcolor += (255,)
        cv2.rectangle(frame, tuple((center - r).astype(int)), tuple((center + r).astype(int)), color=bcolor, thickness=1)

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

    def create_skeleton_video(self, skeleton_json, out_path):
        fps, length, (width, height), kp, c, pids = self.get_video_info(None, skeleton_json)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for i in tqdm(range(length), desc="Writing video result"):
            if i < len(kp):
                skel_frame = self.draw_skeletons(np.zeros((height, width, 3), dtype=np.uint8), kp[i], c[i], (width, height), pids[i])
                out.write(skel_frame)
        out.release()

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
            if ret:
                frame = self.draw_skeletons(frame, kp[i], c[i], (width, height), pids[i])
                out.write(frame)
            else:
                break
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
    skel_root = r'D:\datasets\autism_center_take2\skeletons'
    vids_root = r'D:\datasets\autism_center_take2\segmented_videos'
    out_dir = r'D:\datasets\autism_center_take2\new_videos'
    v = BaseVisualizer()
    for file in os.listdir(skel_root):
        name, ext = path.splitext(file)
        v_name = f'{name}.avi' if path.exists(path.join(vids_root, f'{name}.avi')) else f'{name}.mp4'
        skel = read_json(path.join(skel_root, file))
        v.make_video(path.join(vids_root, v_name), skel, path.join(out_dir, v_name), is_normalized=False, display_pid=True)
