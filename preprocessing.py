import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA 

class PrematureDataset(Dataset):
    def __init__(self, csv_path, n_components):
        self.ehg_sequence = pd.read_csv(csv_path)

        self.X, self.y = self.convert_pca_tensor(n_components) 
        # self.X, self.y = self.convert_tensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def apply_pca(self, n_components):
        pca = PCA(n_components = n_components)
        pca.fit(self.ehg_sequence[["channel_1_filt_0.34_1_hz", "channel_2_filt_0.34_1_hz", "channel_3_filt_0.34_1_hz"]])
        data_pca = pca.transform(self.ehg_sequence[["channel_1_filt_0.34_1_hz", "channel_2_filt_0.34_1_hz", "channel_3_filt_0.34_1_hz"]])
        df_pca = pd.DataFrame(data_pca)
        id_pca = pd.DataFrame({"id":self.ehg_sequence['rec_id'],
                               "pca":df_pca.iloc[:,0],
                               "label":self.ehg_sequence['premature']})
        return id_pca
    def convert_pca_tensor(self, n_components):
        id_pca = self.apply_pca(n_components)
        unique_ids = id_pca['id'].unique()

        sequences = []
        labels = []
        for unique_id in unique_ids:
            id_values = id_pca[id_pca['id'] == unique_id][['pca']].values
            id_label = np.unique(id_pca[id_pca['id'] == unique_id]['label'].values)

            sequences.append(id_values)
            labels.append(id_label)

        sequences_tensor = torch.tensor(sequences)
        labels_tensor = torch.tensor(labels).squeeze()

        return sequences_tensor, labels_tensor
    

    def convert_tensor(self):
        unique_ids = self.ehg_sequence['rec_id'].unique()

        sequences = []
        labels = []
        for unique_id in unique_ids:
            id_values = self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id][["channel_1_filt_0.34_1_hz", "channel_2_filt_0.34_1_hz", "channel_3_filt_0.34_1_hz"]].values
            id_label = np.unique(self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id]['premature'].values)

            sequences.append(id_values)
            labels.append(id_label)

        sequences_tensor = torch.tensor(sequences)
        labels_tensor = torch.tensor(labels).squeeze()

        return sequences_tensor, labels_tensor
    
    
def train_val_test_split(dataset, test_size):
    trainval, test = torch.utils.data.random_split(dataset, [1-test_size, test_size])
    train, val = torch.utils.data.random_split(trainval, [1-test_size, test_size])
    return train, val, test



class PrematureDataloader():
    def __init__(self, dataset, test_size, batch_size = None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.train, self.val, self.test = self.get_dataloader(test_size)

    def get_dataloader(self, test_size):
        train, val, test = self.split(test_size)

        train_batches = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_batches = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        test_batches = DataLoader(test, batch_size=self.batch_size, shuffle=True)

        return train_batches, val_batches, test_batches

class UCRDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep = "\t", header = None)

        self.X, self.y = self.convert_tensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def convert_tensor(self):
        y = self.data.iloc[:,0]
        y_tensor = torch.tensor(y)-1
        self.target_size = len(y_tensor.unique())

        X_df = self.data.drop(0, axis = "columns")
        X_tensor = torch.tensor(X_df.values)
        
        return X_tensor, y_tensor

