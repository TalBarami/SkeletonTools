import os
from os import path

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skeleton_tools.pipe_components.yolo_v5_child_detector import ChildDetector
from skeleton_tools.utils.constants import COLORS, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_json


class Visualizer:
    def draw_json_skeletons(self, frame, skeletons, resolution, display_pid=True, display_bbox=True,
                            is_normalized=False, is_centralized=False, blur_face=False, visualize_confidence=False):
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

        for i, s in enumerate(skeletons):
            pid = int(s['person_id']) if 'person_id' in s.keys() and not s['person_id'] == [-1] else i
            color = tuple(reversed(COLORS[(pid % len(COLORS))]['value']))
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys() and s[src['name']]]:
                x = np.array(s[src['name']][0::2])
                y = np.array(s[src['name']][1::2])
                if is_centralized:
                    x += 0.5
                    y += 0.5
                if is_normalized:
                    x *= (0.8 * width)
                    y *= (0.8 * height)
                x = x.astype(int)
                y = y.astype(int)
                c = np.array(s[f'{src["name"]}_score'])
                pose = list(zip(x, y))
                img = self.draw_skeleton(img, pose, c, src['layout'], color, visualize_confidence=visualize_confidence)

                if src['name'] == 'pose':
                    pose = np.array(pose).T
                    if display_pid:
                        x = pose[0][c > EPSILON]
                        y = pose[1][c > EPSILON]
                        x_center = x.mean() * 0.975
                        y_center = y.min() * 0.9
                        cv2.putText(img, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
                    if display_bbox:
                        self.draw_bbox(img, bounding_box(pose, c))
        return img

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

    def draw_skeleton(self, frame, pose, score, skeleton_layout, color=None, visualize_confidence=False, epsilon=0.05):
        img = np.copy(frame)
        if color is None:
            color = (0, 0, 255)
        for (v1, v2) in skeleton_layout.pairs():
            if score[v1] > epsilon and score[v2] > epsilon:
                cv2.line(img, tuple(pose[v1]), tuple(pose[v2]), color, thickness=2, lineType=cv2.LINE_AA)
        for i, (x, y) in enumerate(pose):
            if score[i] > epsilon:
                jcolor = tuple(np.array(plt.cm.jet(score[i])) * 255) if visualize_confidence else (0, 60, 255)
                joint_size = 4 if visualize_confidence else 2
                cv2.circle(img, (x, y), joint_size, jcolor, thickness=2)
        return img

    def make_skeleton_video(self, skeleton, dst_file, display_pid=False, display_bbox=False, is_normalized=False, is_centralized=False, visualize_confidence=False):
        width, height, length, fps = 1024, 1280, len(skeleton), 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dst_file, fourcc, fps, (height, width))
        for i in tqdm(range(length), desc="Writing video result"):
            frame = np.zeros((width, height, 3), dtype='uint8')
            frame = self.draw_json_skeletons(frame, skeleton[i]['skeleton'], (width, height),
                                             display_pid=display_pid, display_bbox=display_bbox, is_normalized=is_normalized, is_centralized=is_centralized, visualize_confidence=visualize_confidence)
            out.write(frame)
        out.release()


    def make_video(self, video_path, skeleton, dst_file, delay=0, from_frame=None, to_frame=None, display_pid=False, display_bbox=False, is_normalized=False, is_centralized=False, pid_colors=True, blur_faces=False):
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
            while cap.isOpened() and curr_frame < len(skeleton):
                ret, frame = cap.read()
                if ret:
                    if curr_frame >= delay and curr_frame >= from_frame and curr_frame < to_frame:
                        frame = self.draw_json_skeletons(frame, skeleton[curr_frame - delay]['skeleton'], (width, height),
                                                         is_normalized=is_normalized, is_centralized=is_centralized, display_pid=display_pid, display_bbox=display_bbox, pid_colors=pid_colors, blur_face=blur_faces)
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

if __name__ == '__main__':
    org_root = r'D:\datasets\autism_center\skeletons\data'
    skel_root = r'D:\datasets\autism_center\skeletons_filtered\data'
    vids_root = r'D:\datasets\autism_center\segmented_videos'
    out_dir = r'D:\datasets\autism_center\qa_vids'
    bbox_root = r'C:\research\yolov5\runs\detect\100971247_Linguistic_210218_1109_3_Hand flapping_24965_25177\labels'
    v = Visualizer()

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

    for file in os.listdir(skel_root):
        name, ext = path.splitext(file)
        v_name = f'{name}.avi' if path.exists(path.join(vids_root, f'{name}.avi')) else f'{name}.mp4'
        skel = read_json(path.join(skel_root, file))['data']


        v.make_video(path.join(vids_root, v_name), skel, path.join(out_dir, v_name), is_normalized=True)
