import os
import shutil
from argparse import ArgumentParser
from os import path as osp
from pathlib import Path

import pandas as pd

def file_info(f):
    split = f.split('_')
    cid, assessment, group, date, time, camera = split
    return cid, assessment, group, date, time, camera

def get_data_from_pc(src_root, dst_root, table_file):
    r"""Copy files from PC to removable disk. Must support agreement table.
    Args:
        src_root (str): Directory to scan.
        dst_root (str): Destination directory.
        table_file (str): A csv file with 'patient_key', 'redcap_repeat_instance', 'redcap_repeat instrument' - Child id, Date, Assesment respectively.
    """
    df = pd.read_csv(table_file)
    df['assessment'] = df['redcap_repeat_instrument'].apply(lambda a: "ADOS" if 'ados' in a else "PLS" if "pls" in a else "Cognitive")
    df['date'] = df['redcap_repeat_instance'].dt.strftime('%d%m%y')
    df['child_id'] = df['patient_key'].astype(str)
    def agreed(f):
        cid, assessment, _, date, _, _ = file_info(f)
        return df[(df['child_id'] == cid) & (df['assessment'] == assessment) & (df['date'] == date)].empty

    for root, dirs, files in os.walk(src_root):
        files = [f for f in files if osp.splitext(f.lower())[1] in ['.avi', 'mp4'] and agreed(f)]
        for f in files:
            src = osp.join(root, f)
            dst = osp.join(dst_root, f)
            # shutil.copyfile(src, dst)
            print(f'{src}\n---->\n{dst}')

def add_data_to_storage(src_root, dst_root):
    r"""Copy files from removable disk to our main storage.
    Args:
        src_root (str): Directory to scan.
        dst_root (str): Storage source root.
    """
    for root, dirs, files in os.walk(src_root):
        files = [f for f in files if osp.splitext(f.lower())[1] in ['.avi', 'mp4']]
        for f in files:
            cid, assessment, group, date, time, camera = file_info(f)
            src = osp.join(root, f)
            dst = osp.join(dst_root, group, cid, f'{cid}_{assessment}_{group}_{date}', f)
            # Path(dst).mkdir(parents=True, exist_ok=True)
            # shutil.copyfile(src, dst)
            print(f'{src}\n---->\n{dst}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-s", "--src", type=str)
    parser.add_argument("-d", "--dst", type=str)
    parser.add_argument("-t", "--table", type=str, default='')
    args = vars(parser.parse_args())

    mode = args['mode']
    if mode == 'get':
        get_data_from_pc(args['src'], args['dst'], args['table'])
    else:
        add_data_to_storage(args['src'], args['dst'])

# Instructions:
# 1. Install python 3.6 <
# 2. Run: pip install pandas
# 3. Go to the directory of "autism_center_data_storage.py" and run either:
#    > python autism_center_data_storage.py -m get -s "C:/users/recordings" -d "F:/recordings_backup" -t "F:/aggrement_table.csv"
#    > python autism_center_data_storage.py -m add -s "F:/recordings_backup" -d "Z:/recordings"