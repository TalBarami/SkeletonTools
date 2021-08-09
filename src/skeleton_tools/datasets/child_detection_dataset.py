import ast
import os
import random
import shutil
from os import path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skeleton_tools.skeleton_visualization.visualizer import Visualizer
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_json, init_directories


class DataSampler:
    def __init__(self, videos_path, skeletons_path, tagged_df, out_path):
        self.videos_path = videos_path
        self.skeletons_path = skeletons_path
        self.tagged_df = tagged_df
        self.out_path = out_path
        self.img_dir, self.box_dir, self.train_dir, self.test_dir = path.join(self.out_path, 'raw', 'image'), \
                                                                    path.join(self.out_path, 'raw', 'image_box'), \
                                                                    path.join(self.out_path, 'data', 'train'),\
                                                                    path.join(self.out_path, 'data', 'test')
        self.df = pd.read_csv(path.join(self.out_path, 'raw', 'data.csv')) if path.exists(path.join(self.out_path, 'raw', 'data.csv')) else None

        init_directories(path.join(self.img_dir),
                         path.join(self.box_dir),
                         path.join(self.train_dir, 'images'),
                         path.join(self.train_dir, 'labels'),
                         path.join(self.test_dir, 'images'),
                         path.join(self.test_dir, 'labels'))

    def _sample(self, records, n):
        video_files = os.listdir(self.videos_path)
        i = 0
        v = Visualizer()
        df = pd.DataFrame(columns=['image_name', 'center_x', 'center_y', 'height', 'width', 'label', 'org_height', 'org_width'])
        while i < n:
            video_name, segment_name, start_time, end_time, start_frame, end_frame, status, action, child_ids, time, notes = records.sample(n=1).values[0]
            child_ids = ast.literal_eval(child_ids)
            cap = cv2.VideoCapture(path.join(self.videos_path, [f for f in video_files if segment_name in f][0]))
            json_data = read_json(path.join(self.skeletons_path, f'{segment_name}.json'))

            f = random.randint(0, len(json_data['data']) - 1)
            if len(json_data['data']) == 0:
                print(f'Encountered empty json for: {video_name}')
                continue
            if f'{segment_name}_{f}' in df['image_name']:
                print(f'Segment {segment_name}_{f} already in data. Continue.')
                continue

            c = 0
            ret, frame = cap.read()
            while c < f:
                ret, frame = cap.read()
                c += 1
            try:
                if ret and 'skeleton' in json_data['data'][f].keys():
                    cv2.imwrite(path.join(self.img_dir, f'{segment_name}_{f}.png'), cv2.resize(frame, (640, 640)))
                    resolution = np.flip(frame.shape[:2])
                    for skeleton in json_data['data'][f]['skeleton']:
                        pose = (np.array([skeleton['pose'][::2], skeleton['pose'][1::2]]).T * resolution).T.astype(int)
                        box = bounding_box(pose, np.array(skeleton['score']))
                        (cX, cY), (w, h) = box
                        v.draw_bbox(frame, box)
                        is_child = skeleton['person_id'] in child_ids
                        df.loc[df.shape[0]] = [f'{segment_name}_{f}', cX, cY, h, w, is_child, frame.shape[0], frame.shape[1]]
                        cv2.putText(frame, 'Child' if is_child else 'Adult', (cX - 90, cY - h - 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                    cv2.imwrite(path.join(self.box_dir, f'{segment_name}_{f}.png'), cv2.resize(frame, (640, 640)))
                    i += 1
            except ValueError as v:
                print(f'Error in {segment_name}_{f}. Continue.')
            finally:
                cap.release()
        self.df = df
        return df

    def prepare_labels(self, data_dir):
        image_dir, label_dir = path.join(data_dir, 'images'), path.join(data_dir, 'labels')
        for img in os.listdir(image_dir):
            basename, ext = path.splitext(img)
            data = self.df[self.df['image_name'] == basename]

            content = ''
            for i, (name, cx, cy, bh, bw, l, h, w) in data.iterrows():
                content += f'{int(l)} {cx / w} {cy / h} {bw / w} {bh / h}\n'
            with open(path.join(label_dir, f'{basename}.txt'), 'w') as f:
                f.write(content)

    def split_sets(self):
        train, test = train_test_split(os.listdir(self.img_dir))

        for f in train:
            shutil.copy(path.join(self.img_dir, f), path.join(self.train_dir, 'images', f))
        for f in test:
            shutil.copy(path.join(self.img_dir, f), path.join(self.test_dir, 'images', f))

    def init_dataset(self, n):
        if self.df is None:
            samples = self._sample(self.tagged_df[(self.tagged_df['status'] == 'Status.OK') & (self.tagged_df['child_ids'] != "[-1]")], n)
            samples.to_csv(path.join(self.out_path, 'raw', 'data.csv'), index=False)
        self.split_sets()
        self.prepare_labels(self.train_dir)
        self.prepare_labels(self.test_dir)


def main():
    videos_path = r'D:\datasets\autism_center\segmented_videos'
    skeletons_path = r'D:\datasets\autism_center\skeletons\data'
    tagged_df = pd.read_csv(r'D:\datasets\autism_center\qa_dfs\merged.csv')
    out_path = r'D:\datasets\child_detector'

    ds = DataSampler(videos_path, skeletons_path, tagged_df, out_path)
    ds.init_dataset(15000)

if __name__ == '__main__':
    main()
