import os
import pickle

import numpy as np
import pytorch_lightning as pl
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.utils import data
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torchsummary import summary
from torchviz import make_dot
from torch.autograd import Variable

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

    def __init__(self, encoder_units, smaller_units_features_size, n_features_input):
        super().__init__()

        self.lstm_units = ModuleList()

        for count in range(encoder_units, 0, -1):
            lstm_unit = torch.nn.LSTM(
                input_size=n_features_input,
                hidden_size=2 ** (count - 1) * smaller_units_features_size,  ## logaritmic decrese the size
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            self.lstm_units.append(lstm_unit)
            ##The output on torch documentation is in the form ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)
            n_features_input = 2 * 2 ** (count - 1) * smaller_units_features_size

    def forward(self, x):

        hidden_n = None
        for unit in self.lstm_units:
            x, (hidden_n, cells_n) = unit(x)
        return x, hidden_n  ## x output = ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)
        ## hidden_n = (bidirectional=2, Batch_size, Hidden_size)


class Decoder(pl.LightningModule):

    def __init__(self, encoder_units, smaller_units_features_size, n_features_input, n_features_output):
        super().__init__()
        self.lstm_units = ModuleList()
        self.encoder_units = encoder_units
        self.hidden_output_size = 2**encoder_units* smaller_units_features_size ## 2(bidirectional)* 2^encoder units * hidden size
        for count in range(1, encoder_units + 1):
            lstm_unit = torch.nn.LSTM(
                input_size=n_features_input,  ### 2* beacause the encoder is bidirectional
                hidden_size=2 ** (count - 1) * smaller_units_features_size,  ## logaritmic decrese the size
                num_layers=1,
                dropout=0.2,
                bidirectional=True,
                batch_first=True
            )
            self.lstm_units.append(lstm_unit)
            n_features_input = 2 * 2 ** (count - 1) * smaller_units_features_size
            ##The output on torch documentation is in the form ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)

        self.tdd = TimeDistributed(torch.nn.Linear(2 ** encoder_units * smaller_units_features_size, n_features_output),
                                   batch_first=True)
        #self.output_layer = torch.nn.Linear(2 ** encoder_units * smaller_units_features_size, n_features_output)

    def forward(self, x):
        hidden_n = None
        for unit in self.lstm_units:
            x, (hidden_n, cells_n) = unit(x)
        output = self.tdd(x) ##NOTE we delete the linear layer
        return output, x, hidden_n ## x output = ( Batch Size, Sequence Len, D=2 (if bidirectional)*Hidden Size)


class LSTM_AE_ATTENTION(pl.LightningModule):
    ## attention_type = bahdanau / luong
    ## alignment_type = dot_score / general_score / concat_score / latte_score
    def __init__(self, learning_rate, sequence_lenght, n_features_per_sample, n_features_hidden, attention_type="bahdanau", score_type="latte_score", onlinelog=True):
        super(LSTM_AE_ATTENTION, self).__init__()
        self.save_hyperparameters()
        self.onlinelog = onlinelog
        self.learning_rate = learning_rate
        self.attention_type = attention_type
        self.score_type = score_type

        self.seq_len = sequence_lenght

        self.hidden_size = n_features_hidden

        self.encoder = Encoder(encoder_units=2,
                               smaller_units_features_size=n_features_hidden,
                               n_features_input=n_features_per_sample)
        self.decoder = Decoder(encoder_units=2,
                               smaller_units_features_size=n_features_hidden,
                               n_features_input=2 * n_features_hidden,  ## because bidiriectional encoder
                               n_features_output=n_features_per_sample)

        self.score_concat_1 = torch.nn.Sequential(torch.nn.Linear(2* n_features_hidden + self.decoder.hidden_output_size, 1),
                                             torch.nn.Tanh())

        self.alignment = torch.nn.Softmax(dim=1)
        self.general_score = torch.nn.Linear(2*n_features_hidden, 2*n_features_hidden)
        self.concat_score = torch.nn.Sequential(
                                                torch.nn.Linear(2*n_features_hidden*2, 2*n_features_hidden*2),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(2*n_features_hidden*2, 1)
                                            )
        #self.output_layer = torch.nn.Linear(2 ** 2 * n_features_hidden, n_features_per_sample)

    ##################################################################################
    ##
    ##  TRAINING STEPS
    ##
    ##################################################################################
    def training_step(self, batch, batch_idx):
        x = batch
        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        loss = loss / batch.shape[0]
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        final_loss = sum(output['loss'] for output in outs) / len(outs)

        self.log('train_loss_epoch', final_loss)

        wandb.log({"train_loss_epoch": final_loss}) if self.onlinelog else None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, label = batch
        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        return loss, label
    def validation_step(self, batch, batch_idx):
        x = batch

        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        loss = loss / batch.shape[0]
        self.log("val_loss", loss)

        return loss

    def validation_epoch_end(self, outs):
        final_loss = sum(output for output in outs) / len(outs)

        self.log('val_loss_epoch', final_loss)
        wandb.log({"val_loss_epoch": final_loss}) if self.onlinelog else None


    ##################################################################################
    ##
    ##  ATTENTION IMPLEMENTATION
    ##
    ##################################################################################


    '''
        "Neural Machine Translation by jointly learning to align and translate", Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, Sep 2014
    '''
    def _bahdanau_attention(self, H, batchSize):
        # H ( 1, 140, 128) 64*2
        Context_matrix = []

        s_i = torch.zeros((batchSize, 1, self.decoder.hidden_output_size), device=self.device, dtype=torch.float32)

        for i in range(H.shape[1]):
            concat = torch.cat([s_i.expand(-1, H.shape[1], -1), H], dim=2)
            score_vector = self.score_concat_1(concat)
            alignment_vector = self.alignment(score_vector)
            context_vector_i = torch.bmm(torch.transpose(alignment_vector, dim0=1, dim1=2), H)
            o, o_i, s_i = self.decoder(context_vector_i)

            s_i = s_i.permute([1, 0, 2])
            s_i = s_i.reshape(batchSize, 1, self.decoder.hidden_output_size)
            Context_matrix.append(context_vector_i)

        Context_matrix = torch.cat(Context_matrix, dim=1)
        return Context_matrix

    '''
        "LATTE: LSTM Self-Attention based Anomaly Detection in Embedded Automotive Platforms", Vipin K. Kukkala, Sooryaa V. Thiruloga, Sudeep Pasricha, Jul 2021
        
        Score function used in this paper
    '''
    def _latte_score(self, actual_hidden_vector, H):
        return torch.inner(self.general_score(actual_hidden_vector), H)

    '''
        "Effective Approches to Attention-based Neural Machine Translation", Minh-Thang Luong, Hieu Pham, Christopher D. Manning,  Aug 2015 
        
        Score functions
    '''
    def _dot_score(self, H, H_t ):
        return torch.inner(H, H_t)
    def _general_score(self, H, H_t):
        linear = self.general_score(H)
        return torch.bmm(linear, H_t)
    def _concat_score(self, H, H_t):
        concat = torch.cat([H_t, H], dim=2)
        partial = self.concat_score(concat).reshape(1, 1, H.shape[1])
        return partial



    def _self_attention(self, H):
        score_vector = None
        #if self.score_type == "latte_score":
        #    score_vector = self._latte_score(H, torch.transpose(H, dim0=1, dim1=2))
        if self.score_type == "dot_score":
            score_vector = torch.bmm(H, torch.transpose(H, dim0=1, dim1=2))#self._dot_score(H, torch.transpose(H, dim0=1, dim1=2))
        elif self.score_type == "general_score":
            score_vector = self._general_score(H, torch.transpose(H, dim0=1, dim1=2))
        #elif self.score_type == "concat_score":
        #    score_vector = self._concat_score(H, torch.transpose(H, dim0=1, dim1=2))

        alignment_vector = self.alignment(score_vector)
        context_matrix = torch.bmm(alignment_vector, H)
        return context_matrix


    def _extractHiddenStatesfromTimeSeries(self, x):

        batchSize = x.shape[0]

        ## x: (1, 140, 1)
        ##Input is in the shape (BatchSize, SequenceLen, NinputFeatures)
        H = []
        for i in range(x.shape[1]):
            input = x[:, i, ...].reshape(batchSize, 1, x.shape[2])
            o, hidden = self.encoder(input)
            hidden = hidden.permute([1, 0, 2])
            hiddensize = hidden.shape[2]*2 ## because bidirectional
            hidden = hidden.reshape((batchSize, 1, hiddensize))
            H.append(hidden)

        H = torch.cat(H, dim=1)
        #H(1, 140, 128)
        return H
    def forward(self, x):
        batchSize = x.shape[0]
        '''
            (1) We need Encoder(X) -> [h1, h2, ...., hT] = H
        '''
        H = self._extractHiddenStatesfromTimeSeries(x)
        H_star = self._bahdanau_attention(H, batchSize) if self.attention_type == "bahdanau" else self._self_attention(H)
        Output, _, _ = self.decoder(H_star)

        return Output
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



