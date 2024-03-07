import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class PrematureDataset(Dataset):
    def __init__(self, csv_path):
        self.ehg_sequence = pd.read_csv(csv_path)

        self.X, self.y = self.convert_tensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def convert_tensor(self):
        unique_ids = self.ehg_sequence['rec_id'].unique()

        sequences = []
        labels = []
        for unique_id in unique_ids:
            id_values = self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id]["channel_1_filt_0.34_1_hz"].values
            id_label = np.unique(self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id]['premature'].values)

            sequences.append(id_values)
            labels.append(id_label)

        sequences_tensor = torch.tensor(sequences)
        labels_tensor = torch.tensor(labels).squeeze()

        return sequences_tensor, labels_tensor


class TrainValTest():
    def __init__(self, dataset, test_size, batch_size = None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.train, self.val, self.test = self.get_dataloader(test_size)

    def split(self, test_size):
        trainval, test = torch.utils.data.random_split(self.dataset, [1-test_size, test_size])
        train, val = torch.utils.data.random_split(trainval, [1-test_size, test_size])
        return train, val, test

    def get_dataloader(self, test_size):
        train, val, test = self.split(test_size)

        train_batches = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_batches = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        test_batches = DataLoader(test, batch_size=self.batch_size, shuffle=True)

        return train_batches, val_batches, test_batches

