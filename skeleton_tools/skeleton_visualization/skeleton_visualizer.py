from abc import ABC, abstractmethod

from os import path as osp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skeleton_tools.skeleton_visualization.draw_utils import blur_area, draw_pid, draw_bbox
from skeleton_tools.utils.constants import COLORS, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box


class SkeletonVisualizer(ABC):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, display_child_bbox=False):
        self.graph_layout = graph_layout
        self.display_pid = display_pid
        self.display_bbox = display_bbox
        self.denormalize = int(denormalize)
        self.decentralize = decentralize
        self.blur_face = blur_face
        self.display_child_bbox = display_child_bbox

    def draw_skeletons(self, frame, skeletons, scores, epsilon=0.25, thickness=5, resolution=None, pids=None, child_id=None, child_box=None, detection_conf=None):
        img = np.copy(frame)

        if pids is None:
            pids = np.arange(skeletons.shape[0])

        if self.denormalize:
            skeletons *= resolution
        skeletons = skeletons.astype(int)

        if self.blur_face:
            for i, (pose, score) in enumerate(zip(skeletons, scores)):
                face_joints = [k for k, v in self.graph_layout.joints().items() if any([s in v for s in self.graph_layout.face_joints()])]
                facebox = bounding_box(pose[face_joints].T, score[face_joints]).astype(int)
                img = blur_area(img, facebox[:2], facebox[2:].max())

        for lst_id, (pose, score, pid) in enumerate(zip(skeletons, scores, pids)):
            if child_id is None:
                color = tuple(reversed(COLORS[(pid % len(COLORS))]['value']))
            else:
                color = (0, 0, 255) if child_id == lst_id else (255, 0, 0)
            if img.shape[-1] > 3:
                color += (255,)

            img = self.draw_skeleton(img, pose, score, thickness=thickness, edge_color=color, epsilon=epsilon)

            if self.display_pid:
                draw_pid(img, pose.T, score, pid, color)

            if self.display_bbox:
                draw_bbox(img, bounding_box(pose.T, score, epsilon))

        if self.display_child_bbox and child_box is not None and detection_conf > 0:
            child_box = child_box.astype(int)
            center, r = child_box[:2], child_box[2:4]
            draw_bbox(img, (center, r), bcolor=(0, 0, 255))
            cv2.putText(img, str(np.round(detection_conf, 3)), (int(center[0] - r[0] // 2), int(center[1] - r[1] // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return img

    def draw_skeleton(self, frame, pose, score, edge_color=None, thickness=5, epsilon=0.05):
        img = np.copy(frame)
        if edge_color is None:
            edge_color = (0, 0, 255)
        joint_color = (0, 60, 255)
        if img.shape[-1] > 3:
            joint_color += (255,)
        for (v1, v2) in self.graph_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(img, tuple(pose[v1]), tuple(pose[v2]), edge_color, thickness=thickness, lineType=cv2.LINE_AA)
        # for i, (x, y) in enumerate(pose):
        #     if score[i] > epsilon:
        #         jcolor, jsize = (tuple(np.array(plt.cm.jet(score[i])) * 255), 4) if self.show_confidence else (joint_color, 2)
        #         cv2.circle(img, (x, y), jsize, jcolor, thickness=2)
        return img

    def create_skeleton_video(self, skeleton_json, out_path):
        fps, length, (width, height), kp, c, pids = self.get_video_info(None, skeleton_json)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for i in tqdm(range(length), desc="Writing video result"):
            if i < len(kp):
                skel_frame = self.draw_skeletons(np.zeros((height, width, 3), dtype=np.uint8), kp[i], c[i], resolution=(width, height), pids=pids[i])
                out.write(skel_frame)
        out.release()

    def create_double_frame_video(self, video_path, skeleton_data, out_path, start=None, end=None):
        fps, length, (width, height), kp, c, pids, child_ids, detections, child_bbox, adjust = self.get_video_info(video_path, skeleton_data)
        start = int(0 if start is None else start * fps)
        end = int(length if end is None else end * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (2 * width, height))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in tqdm(range(start, end), desc="Writing video result"):
            ret, frame = cap.read()
            skel_frame = np.zeros_like(frame)
            if child_ids is not None:
                skel_frame = self.draw_skeletons(skel_frame, kp[i], c[i], resolution=(width, height), pids=pids[i], child_id=child_ids[i], child_box=child_bbox[i], detection_conf=detections[i])
            else:
                skel_frame = self.draw_skeletons(skel_frame, kp[i], c[i], resolution=(width, height), pids=pids[i])
            out.write(np.concatenate((frame, skel_frame), axis=1))
        cap.release()
        out.release()

    def create_video(self, video_path, skeleton_data, out_path):
        fps, length, (width, height), kp, c, pids, child_ids, detections, child_bbox, adjust = self.get_video_info(video_path, skeleton_data)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        cap = cv2.VideoCapture(video_path)

        for j in tqdm(range(length), desc="Writing video result"):
            ret, frame = cap.read()
            if ret:
                if j < adjust:
                    continue
                i = j - adjust
                if child_ids is not None:
                    frame = self.draw_skeletons(frame, kp[i], c[i], resolution=(width, height), pids=pids[i], child_id=child_ids[i], child_box=child_bbox[i], detection_conf=detections[i])
                else:
                    frame = self.draw_skeletons(frame, kp[i], c[i], resolution=(width, height), pids=pids[i])
                out.write(frame)
            else:
                break
        cap.release()
        out.release()

    def export_frames(self, video_path, skeleton, out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        fps, length, (width, height), kp, c, pids, child_ids, detections, child_bbox, adjust = self.get_video_info(video_path, skeleton)
        cap = cv2.VideoCapture(video_path)

        for i in tqdm(range(length), desc="Writing video result"):
            ret, frame = cap.read()
            if child_ids is not None:
                frame = self.draw_skeletons(frame, kp[i], c[i], resolution=(width, height), pids=pids[i], child_id=child_ids[i], child_box=child_bbox[i], detection_conf=detections[i])
            else:
                frame = self.draw_skeletons(frame, kp[i], c[i], resolution=(width, height), pids=pids[i])
            cv2.imwrite(osp.join(out_path, f'{i}.png'), frame)
        cap.release()

    def to_image(self, frame):
        top, bottom, left, right = [20] * 4
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128,128,128))
        frame[:, :, 3] = (255 * np.where((frame[:, :, :3] != 255).any(axis=2), 1, 0.6)).astype(np.uint8)
        return frame

    def sample_frames(self, video_path, skeleton_data, frame_idxs, out_dir):
        fps, frames_count, (width, height), kp, c, pids, child_ids, detections, boxes = self.get_video_info(video_path, skeleton_data)
        cap = cv2.VideoCapture(video_path)
        white = np.ones((height, width, 4)) * 255

        for i in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = np.concatenate((frame, np.zeros((height, width, 1))), axis=2)
            frame[:, :, 3] = (255 * (frame[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
            if ret:
                skeleton = self.draw_skeletons(np.copy(white), kp[i], c[i], pids=np.ones(kp.shape[0], dtype='uint8') * 0)
                skeleton = self.to_image(skeleton)
                skeleton_child_detect = self.draw_skeletons(np.copy(white), kp[i], c[i], child_id=child_ids[i])
                skeleton_child_detect = self.to_image(skeleton_child_detect)
                cv2.imwrite(osp.join(out_dir, f'org_{i}.png'), frame)
                cv2.imwrite(osp.join(out_dir, f'skeleton_{i}.png'), skeleton)
                cv2.imwrite(osp.join(out_dir, f'skeleton_cd_{i}.png'), skeleton_child_detect)


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
