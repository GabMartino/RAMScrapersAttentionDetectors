import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch.utils.data as data




class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, good_data, bad_data):

        self.good_data = good_data
        self.good_label = torch.zeros(self.good_data.shape[0])

        if bad_data is not None:
            self.bad_data = bad_data
            self.bad_label = torch.ones(self.bad_data.shape[0])

            self.data = torch.cat([self.good_data, self.bad_data], dim=0)
            self.labels = torch.cat([self.good_label, self.bad_label], dim=0)
        else:
            self.data = self.good_data
            self.labels = self.good_label
        #self.data = self.bad_data
        print(self.good_data.shape[0])
        print(self.bad_data.shape[0])
        #self.labels = self.bad_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        object = self.data[idx]
        label = self.labels[idx]
        return object, label
class POSTransactionDataLoader(pl.LightningDataModule):


    def __init__(self, normal_data_path, abnormal_data_path, batch_size, validation_percent=0.2):
        super().__init__()
        self.batch_size = batch_size

        self.normal_data = None
        if normal_data_path is not None:
            with open(normal_data_path, "rb") as f:
                self.normal_data = pickle.load(f)


            self.normal_data = np.asarray(self.normal_data, dtype=np.float32)
            self.normal_data = [torch.tensor(s).float() for s in self.normal_data]
            self.normal_data = torch.stack(self.normal_data, dim=0)

        self.abnormal_data = None
        if abnormal_data_path is not None:
            with open(abnormal_data_path, "rb") as f:
                self.abnormal_data = pickle.load(f)
            self.abnormal_data = np.asarray(self.abnormal_data, dtype=np.float32)
            self.abnormal_data = [torch.tensor(s).float() for s in self.abnormal_data]
            self.abnormal_data = torch.stack(self.abnormal_data, dim=0)

        train_set_size = int(len(self.normal_data) * (1 - validation_percent))
        valid_set_size = len(self.normal_data) - train_set_size
        self.train_data, self.val_data = data.random_split(self.normal_data, [train_set_size, valid_set_size])

        val_data = self.val_data.dataset[self.val_data.indices]
        self.test_dataset = CombinedDataset(val_data, self.abnormal_data)
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, prefetch_factor=os.cpu_count(), num_workers=os.cpu_count(), shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=os.cpu_count())