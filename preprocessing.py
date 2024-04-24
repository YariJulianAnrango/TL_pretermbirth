import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from train_test_split import random_split

class PrematureDataset(Dataset):
    def __init__(self, csv_path, n_components, feature_index):
        data = pd.read_csv(csv_path, index_col = 0)
        self.ehg_sequence = self.normalize(data)
        if n_components < 3:
            self.X, self.y = self.convert_pca_tensor(n_components) 
        else:
            self.X, self.y = self.convert_tensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def apply_pca(self, n_components):
        pca = PCA(n_components = n_components)
        pca.fit(self.ehg_sequence[[0, 1, 2]])
        data_pca = pca.transform(self.ehg_sequence[[0, 1, 2]])
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
            id_values = self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id][[0,1,2]].values
            id_label = np.unique(self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id]['premature'].values)

            sequences.append(id_values)
            labels.append(id_label)

        sequences_tensor = torch.tensor(sequences)
        labels_tensor = torch.tensor(labels).squeeze()

        return sequences_tensor, labels_tensor
    
    def normalize(self, data):
        y = data.loc[:,"premature"]
        id = data.loc[:, "rec_id"]
        x = data.drop(["premature", "rec_id"], axis = "columns").values
  
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        
        scaled_df = pd.DataFrame(scaled)
        scaled_df.insert(0, "premature", y)
        scaled_df.insert(1, "rec_id", id)
        
        return scaled_df  
    
def train_val_split(dataset, test_size):
    train, val = random_split(dataset, [1-test_size, test_size])
    return train, val

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
        # self.data = self.normalize(df)

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
    
    def normalize(self, data):
        y = data.iloc[:,0]
        x = data.drop(0, axis = "columns").values

        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        
        scaled_df = pd.DataFrame(scaled)
        scaled_df.insert(0, "label", y)
        
        return scaled_df

def get_train_val(dataframe):
    unique_ids = dataframe["rec_id"].unique()
    val_size = 0.3
    val_ids = np.random.choice(unique_ids, int(len(unique_ids)*val_size))
    
    val = dataframe[dataframe["rec_id"].isin(val_ids)].reset_index(drop = True)
    train = dataframe.drop(dataframe[dataframe["rec_id"].isin(val_ids)].index).reset_index(drop = True)

    return train, val

class PrematureDatasetSplit(Dataset):
    def __init__(self, dataframe):#, feature_index):
        self.ehg_sequence = self.normalize(dataframe)
        #self.feature_index = feature_index
        self.X, self.y = self.convert_tensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        #X_feature = X[:,self.feature_index]
        return (X[:,0], X[:, 1], X[:, 2]), self.y[idx]
    
    def normalize(self, data):
        y = data.loc[:,"premature"]
        id = data.loc[:, "rec_id"]
        x = data.drop(["premature", "rec_id"], axis = "columns").values
  
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        
        scaled_df = pd.DataFrame(scaled)
        scaled_df.insert(0, "premature", y)
        scaled_df.insert(1, "rec_id", id)
        
        return scaled_df
    
    def convert_tensor(self):
        unique_ids = self.ehg_sequence['rec_id'].unique()

        sequences = []
        labels = []
        for unique_id in unique_ids:
            id_values = self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id][[0,1,2]].values
            id_label = np.unique(self.ehg_sequence[self.ehg_sequence['rec_id'] == unique_id]['premature'].values)

            sequences.append(id_values)
            labels.append(id_label)

        sequences_tensor = torch.tensor(sequences)
        labels_tensor = torch.tensor(labels).squeeze()

        return sequences_tensor, labels_tensor
   