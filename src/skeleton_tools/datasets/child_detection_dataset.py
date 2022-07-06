import ast
import os
import random
import shutil
from os import path as osp
import itertools as it

import cv2
import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.skeleton_visualization.base_visualizer import BaseVisualizer
from skeleton_tools.skeleton_visualization.json_visualizer import JsonVisualizer
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_json, init_directories, write_json


def rotate(l, n=-1):
    l[:] = l[n:] + l[:n]

class DataSampler:
    def __init__(self, files, tagged_df, out_path):
        self.files = files
        self.tagged_df = tagged_df
        self.out_path = out_path
        self.raw_dir = osp.join(self.out_path, 'raw')
        self.img_dir, self.box_dir, self.train_dir, self.test_dir = osp.join(self.raw_dir, 'image'), \
                                                                    osp.join(self.raw_dir, 'image_box'), \
                                                                    osp.join(self.out_path, 'data', 'train'),\
                                                                    osp.join(self.out_path, 'data', 'test')
        self.columns = ['image_name', 'center_x', 'center_y', 'height', 'width', 'label', 'org_height', 'org_width']
        self.df = pd.read_csv(osp.join(self.out_path, 'raw', 'data.csv')) if osp.exists(osp.join(self.out_path, 'raw', 'data.csv'))\
            else pd.DataFrame(columns=self.columns)

        init_directories(osp.join(self.img_dir),
                         osp.join(self.box_dir),
                         osp.join(self.train_dir, 'images'),
                         osp.join(self.train_dir, 'labels'),
                         osp.join(self.test_dir, 'images'),
                         osp.join(self.test_dir, 'labels'))

    def _sample(self, records, n):
        frames_space = 25
        vis = JsonVisualizer(BODY_25_LAYOUT)
        pool_path = osp.join(self.raw_dir, 'frames_pool.json')
        if osp.exists(pool_path):
            pool = read_json(pool_path)
        else:
            files = {k: v for k, v in self.files.items() if k in records['segment_name'].tolist()}
            for vinf in files.values():
                vinf['length'] = len(read_json(vinf['skeleton'])['data'])
            pool = [[(v, s, i) for i in range(frames_space, self.files[s]['length'], frames_space)] for v, s in records[['video', 'segment_name']].values]
            pool = [x for xs in pool for x in xs]
            pool = [[v, [(ss, c) for _, ss, c in s]] for v, s in it.groupby(pool, lambda x: x[0])]
            for vgroup in pool:
                vgroup[1] = [(a, [i for _, i in b]) for a, b in it.groupby(vgroup[1], lambda x: x[0])]
            write_json(pool, osp.join(self.raw_dir, 'frames_pool.json'))
        m = 0
        for video, vgroup in pool:
            for segment, sgroup in vgroup:
                for i in sgroup:
                    if f'{segment}_{i}' in self.df['image_name'].values:
                        sgroup.remove(i)
                random.shuffle(sgroup)
                m += len(sgroup)
                if len(sgroup) == 0:
                    vgroup.remove([segment, sgroup])
            if len(vgroup) == 0:
                pool.remove([video, vgroup])
        print(f'Total {m} candidate samples from {len(pool)} videos.')
        pbar = tqdm(total=n)

        i, j = 0, len(self.df['image_name'].unique())
        pbar.update(j)
        while j < n:
            _, segments = pool[-1]
            segment_name, frames = segments[-1]
            f = frames.pop()


            if len(frames) == 0:
                segments.pop()
            else:
                rotate(segments, 1)
                i += 1
            if len(segments) == 0:
                pool.pop()
            elif i == len(segments):
                i = 0
                rotate(pool, 1)

            child_ids = ast.literal_eval(records[records['segment_name'] == segment_name]['child_ids'].iloc[0])
            file = self.files[segment_name]
            json_data = read_json(file['skeleton'])

            image_name = f'{segment_name}_{f}'
            if image_name in self.df['image_name'].values:
                print(f'Segment {image_name} already in data. Continue.')
                continue

            cap = cv2.VideoCapture(file['video'])
            try:
                c = 0
                ret, frame = cap.read()
                while c < f:
                    ret, frame = cap.read()
                    c += 1
                if ret and 'skeleton' in json_data['data'][f].keys():
                    org_frame = frame.copy()
                    resolution = np.array(frame.shape[:2])
                    for skeleton in json_data['data'][f]['skeleton']:
                        pose = np.array([skeleton['pose'][::2], skeleton['pose'][1::2]]).astype(int)
                        center, size = bounding_box(pose, np.array(skeleton['pose_score']))
                        size = (size * np.array([1.1, 1.3])).astype(int)
                        cX, cY = center.astype(int)
                        w, h = size
                        is_child = int(skeleton['person_id'] in child_ids)
                        color = (0, 255, 0) if is_child else (0, 0, 255)
                        vis.draw_bbox(frame, (center, size), bcolor=color)
                        self.df.loc[self.df.shape[0]] = [image_name, cX, cY, h, w, is_child, resolution[1], resolution[0]]
                        frame = cv2.putText(frame, 'Child' if is_child else 'Adult', (cX - 90, cY - h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=2, lineType=cv2.LINE_AA)

                    cv2.imwrite(osp.join(self.img_dir,f'{image_name}.png'), org_frame)
                    cv2.imwrite(osp.join(self.box_dir, f'{image_name}.png'), frame)
                    j += 1
                    pbar.update(1)
                    self.df.to_csv(osp.join(self.out_path, 'raw', 'data.csv'), index=False)
            except ValueError as v:
                print(f'Error in {segment_name}_{f}. Continue.')
            finally:
                cap.release()

    def prepare_labels(self, data_dir):
        image_dir, label_dir = osp.join(data_dir, 'images'), osp.join(data_dir, 'labels')
        for img in os.listdir(image_dir):
            basename, ext = osp.splitext(img)
            data = self.df[self.df['image_name'] == basename][self.columns]

            content = ''
            for i, (name, cx, cy, bh, bw, l, h, w) in data.iterrows():
                _cx, _cy, _w, _h = min(1, cx/w), min(1, cy/h), min(1, bw/w), min(1, bh/h)
                content += f'{int(l)} {_cx} {_cy} {_w} {_h}\n'
            with open(osp.join(label_dir, f'{basename}.txt'), 'w') as f:
                f.write(content)

    def split_sets(self):
        if len(os.listdir(osp.join(self.train_dir, 'images'))) > 0:
            return
        self.df['video'] = self.df['image_name'].apply(lambda s: '_'.join(s.split('_')[:-4]))
        train_vids, test_vids = train_test_split(self.df['video'].unique(), test_size=0.2)
        train, test = [], []

        for i, row in self.df.iterrows():
            image_name = f'{row["image_name"]}.png'
            if row['video'] in train_vids:
                train.append(image_name)
            else:
                test.append(image_name)

        for f in train:
            shutil.copy(osp.join(self.img_dir, f), osp.join(self.train_dir, 'images', f))
        for f in test:
            shutil.copy(osp.join(self.img_dir, f), osp.join(self.test_dir, 'images', f))

    def init_dataset(self, n):
        if len(self.df['image_name'].unique()) < n:
            records = self.tagged_df
            records = records[(records['status'] == 'Status.OK') &
                              (records['child_ids'] != "[-1]") &
                              (records['segment_name'].isin(self.files.keys()))]
            self._sample(records, n)
        self.split_sets()
        self.prepare_labels(self.train_dir)
        self.prepare_labels(self.test_dir)

    def init_single_person_dataset(self, out_dir):
        images = os.listdir(self.img_dir)
        db = {}
        for img_name in images:
            name, ext = osp.splitext(img_name)
            df = self.df[self.df['image_name'] == name]
            if not df.empty:
                img = cv2.imread(osp.join(self.img_dir, img_name))
                for i, r in df[self.columns].iterrows():
                    cX, cY, w, h = r['center_x'], r['center_y'], r['width'], r['height']
                    sX, sY = cX - w // 2, cY - h // 2
                    sub = img[sX:sX+w, sY:sY+h]
                    out_name = f'{name}_p{i}{ext}'
                    db[out_name] = df['label']
                    cv2.imwrite(osp.join(out_dir, f'{out_name}'), sub)
            write_json(db, osp.join(self.raw_dir, 'single_person.json'))




def main():
    annotations = pd.read_csv(r'D:\datasets\all_data\annotations\multi_label.csv')
    files = read_json(r'D:\datasets\lancet_submission_data\child_detector\raw_data.json')
    # videos_path = r'D:\datasets\autism_center\segmented_videos'
    # skeletons_path = r'D:\datasets\autism_center\skeletons\data'
    # tagged_df = pd.read_csv(r'D:\datasets\autism_center\qa_dfs\merged.csv')
    out_path = r'C:\research\yolov5_dataset'

    ds = DataSampler(files, annotations, out_path)
    ds.init_dataset(20000)

if __name__ == '__main__':
    main()
    # hadas_dor = {'skeletons_dir': r'D:\datasets\taggin_hadas&dor_remake\skeletons',
    #              'videos_dir': r'D:\datasets\taggin_hadas&dor_remake\segmented_videos'}
    # ofri = {'skeletons_dir': r'D:\datasets\tagging_ofri\skeletons',
    #         'videos_dir': r'D:\datasets\tagging_ofri\segmented_videos'}
    # files = {}
    # for ann, db in [('hadas_dor', hadas_dor), ('ofri', ofri)]:
    #     files.update({osp.splitext(f)[0]: {'annotator': ann,
    #                                        'skeleton': osp.join(db['skeletons_dir'], f)} for f in os.listdir(db['skeletons_dir'])})
    #     for file in os.listdir(db['videos_dir']):
    #         name = osp.splitext(file)[0]
    #         if name in files:
    #             files[name]['video'] = osp.join(db['videos_dir'], file)
    # write_json(files, r'D:\datasets\lancet_submission_data\child_detector\raw_data.json')
