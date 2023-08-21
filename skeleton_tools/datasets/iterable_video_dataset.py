from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import cv2
import torch

# class IterableVideoDataset(Dataset):
#     def __init__(self, video_path, batch_size):
#         self.video_path = video_path
#         self.cap = cv2.VideoCapture(self.video_path)
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         batch = []
#         for _ in range(self.batch_size):
#             ret, frame = self.cap.read()
#             if ret:
#                 batch.append(frame)
#                 # self.i += 1
#             elif len(batch) == 0:
#                 raise StopIteration
#             else:
#                 break
#         return batch
#
#     def __del__(self):
#         self.cap.release()


class IterableVideoDataset(IterableDataset):
    def __init__(self, video_path, device='cuda'):
        self.video_path = video_path
        self.cap = None
        self.device = torch.device(device)

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return torch.from_numpy(frame).to(self.device)



if __name__ == '__main__':
    dataset = IterableVideoDataset(r'Z:\Users\TalBarami\models_outputs\videos\666661888_ADOS_Clinical_190218_1220_4_Facial_None_None.mp4')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    for batch in dataloader:
        print(batch.shape)
        break