import torch
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
    def __init__(self, data_numpy, data_info):
        self.data_numpy = torch.tensor(data_numpy)
        self.data_info = data_info

    def __getitem__(self, index):
        x, i = self.data_numpy[index], self.data_info[index]
        return x, i

    def __len__(self):
        return self.data_numpy.shape[0]