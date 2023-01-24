import os
import pickle

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import Sequential, LSTM, LSTMCell
import torch
from torch.utils import data
from torch.nn import ModuleList
from torch.utils.data import DataLoader
import hydra
import matplotlib.pyplot as plt
import wandb


class TimeDistributed(pl.LightningModule):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
class Encoder(pl.LightningModule):

    def __init__(self, encoder_units, smaller_units_features_size, n_features_input, bidirectional):
        super().__init__()

        self.lstm_units = ModuleList()

        for count in range(encoder_units, 0, -1):
            lstm_unit = torch.nn.LSTM(
                input_size=n_features_input,
                hidden_size= 2**(count - 1) * smaller_units_features_size,## logaritmic decrese the size
                num_layers=1,
                bidirectional = bidirectional,
                batch_first=True
                )
            self.lstm_units.append(lstm_unit)
            ##The output on torch documentation is in the form ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)
            n_features_input = 2**(count - 1) * smaller_units_features_size
            n_features_input *= 2 if bidirectional else 1

    def forward(self, x):

        hidden_n = None
        for unit in self.lstm_units:
            x, (hidden_n, cells_n) = unit(x)
        return x, hidden_n ## x output = ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)
                         ## hidden_n = (bidirectional=2, Batch_size, Hidden_size)

class Decoder(pl.LightningModule):

    def __init__(self, encoder_units, smaller_units_features_size, n_features_input, n_features_output, bidirectional):
        super().__init__()
        self.lstm_units = ModuleList()

        for count in range(1, encoder_units + 1):

            lstm_unit = torch.nn.LSTM(
                input_size=n_features_input,### 2* beacause the encoder is bidirectional
                hidden_size=2 ** (count - 1) * smaller_units_features_size,  ## logaritmic decrese the size
                num_layers=1,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_units.append(lstm_unit)
            n_features_input = 2 ** (count - 1) * smaller_units_features_size
            n_features_input *= 2 if bidirectional else 1

        output_size = smaller_units_features_size*(2**(encoder_units))
        print(output_size)
            ##The output on torch documentation is in the form ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)
        self.tdd = torch.nn.Linear(output_size, n_features_output) #TimeDistributed(torch.nn.Linear(2**encoder_units*smaller_units_features_size, n_features_output), batch_first=True)


    def forward(self, x):

        hidden_n = None
        for unit in self.lstm_units:
            x, (hidden_n, cells_n) = unit(x)
        x = self.tdd(x)
        return x## x output = ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)

class LSTM_AE(pl.LightningModule):

    def __init__(self,  learning_rate, sequence_lenght, n_features_per_sample, n_features_hidden, bidirectional, encoder_units= 2, onlinelog=True):
        super(LSTM_AE, self).__init__()
        self.save_hyperparameters()
        self.onlineLog = onlinelog
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.seq_len = sequence_lenght
        #self.train_f1 = torchmetrics.F1Score(task='binary')
        #self.precision = torchmetrics.Precision(task='binary')
        #self.recall = torchmetrics.Recall(task='binary')

        self.encoder = Encoder(encoder_units=encoder_units,
                               smaller_units_features_size=n_features_hidden,
                               n_features_input=n_features_per_sample,
                               bidirectional=bidirectional)
        self.decoder = Decoder(encoder_units=encoder_units,
                               smaller_units_features_size=n_features_hidden,
                               n_features_input= n_features_hidden*2 if bidirectional else n_features_hidden,## because bidiriectional encoder
                               n_features_output=n_features_per_sample,
                               bidirectional=bidirectional)

    def predict_step(self, batch, batch_idx):
        x, label = batch
        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        #loss = loss / batch.shape[0]
        return loss, label


    def training_step(self, batch, batch_idx):
        x = batch
        predicted_x = self.forward(x)
        #x = x.reshape(predicted_x.shape)
        #print(predicted_x, x)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        loss = loss/batch.shape[0]
        self.log("train_loss", loss)
        #wandb.log({"train_loss": loss})
        return loss

    def training_epoch_end(self, outs):
        final_loss = sum(output['loss'] for output in outs) / len(outs)
        self.log('train_loss_epoch', final_loss)
        wandb.log({"train_loss_epoch": final_loss}) if self.onlineLog else None

    def validation_step(self, batch, batch_idx):
        x = batch
        predicted_x = self.forward(x)
        #x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        loss = loss / batch.shape[0]
        self.log("val_loss", loss)
        #wandb.log({"val_loss": loss})
        return loss
    def validation_epoch_end(self, outs):
        final_loss = sum(output for output in outs) / len(outs)
        self.log('val_loss_epoch', final_loss)
        wandb.log({"val_loss_epoch": final_loss}) if self.onlineLog else None

    def test_step(self, batch, batch_idx):
        x = batch
        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        self.log("test_loss", loss)
        return loss


    def forward(self, x):
        batchSize = x.shape[0]
        x, hidden = self.encoder(x) ## hidden is in the shape ( 2 if bidirectional else 1, batch_size, Hidden_size)

        hidden = hidden.permute([1, 0, 2])
        hiddensize = hidden.shape[2]
        hiddensize *= 2 if self.bidirectional else 1
        hidden = hidden.reshape((batchSize, 1, hiddensize))
        hidden = hidden.repeat(1, self.seq_len, 1)
        x = self.decoder(hidden)
        x = torch.flip(x, [1])
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
