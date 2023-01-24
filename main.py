import glob
import os
import pickle

import numpy as np
from omegaconf import OmegaConf
import sklearn
from tqdm import tqdm

import wandb

import hydra
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchsummary import summary

from models.LSTM_AE import LSTM_AE
from models.LSTM_AE_ATTENTION import LSTM_AE_ATTENTION

from DataLoaders.POSTransactionDataLoader import POSTransactionDataLoader

from models.TRANSFORMER import TransformerAE


def printErrorsDistribution(prediction, path, filename, modelName, parameters):
    benign = [error.item() for (error, label) in prediction if label[0] == 0]
    malign = [error.item() for (error, label) in prediction if label[0] == 1]
    #print(len(malign), len(benign))
    from random import sample
    #benign = sample(benign, 10*len(malign))

    plt.hist(benign, bins=250, label="Benign")
    plt.hist(malign, bins=250, label="Malign")
    plt.title("Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path + filename)
    plt.show()



def getModel(cfg, seq_len, n_features_per_elem, modelPath):
    model = None
    if cfg.model == "lstm_ae":
        model = LSTM_AE(learning_rate=cfg.learning_rate,
                        sequence_lenght=seq_len,
                        n_features_per_sample=n_features_per_elem,
                        n_features_hidden=cfg.lstm_ae.hidden_size_smaller_unit,
                        bidirectional=cfg.lstm_ae.bidirectional,
                        onlinelog=cfg.onlinelog)
        modelPath += "LSTM_AE/" + str(cfg.lstm_ae.hidden_size_smaller_unit) + "/" + str(seq_len) + "/"
        parameterTrack = cfg.lstm_ae.hidden_size_smaller_unit
        print(seq_len, model.seq_len)
    elif cfg.model == "lstm_ae_attention":
        model = LSTM_AE_ATTENTION(learning_rate=cfg.learning_rate,
                                  sequence_lenght=seq_len,
                                  n_features_per_sample=n_features_per_elem,
                                  n_features_hidden=cfg.lstm_ae_attention.hidden_size_smaller_unit,
                                  attention_type=cfg.lstm_ae_attention.attention_type,
                                  score_type=cfg.lstm_ae_attention.attention_score_function,
                                  onlinelog=cfg.onlinelog)
        parameterTrack = cfg.lstm_ae_attention.hidden_size_smaller_unit
        modelPath += "LSTM_AE_ATTENTION/" + str(cfg.lstm_ae_attention.attention_type) + "/"
        if str(cfg.lstm_ae_attention.attention_type) != "bahdanau":
            modelPath += str(cfg.lstm_ae_attention.attention_score_function) +"/"
        modelPath += str(cfg.lstm_ae_attention.hidden_size_smaller_unit) +"/" + str(seq_len) + "/"
    elif cfg.model == "transformer":
        model = TransformerAE(dim=cfg.transformer.dim_model,
                              depth=cfg.transformer.depth,
                              seq_len=seq_len,
                              heads=cfg.transformer.num_heads,
                              n_features=n_features_per_elem,
                              learning_rate=cfg.learning_rate
                              )
        parameterTrack = cfg.transformer.dim_model
        modelPath += "TRANSFORMER/" + str(seq_len) + "/" + str(cfg.transformer.dim_model) + "/" + str(cfg.transformer.depth) + "/" + str(cfg.transformer.num_heads) + "/"

    return model, parameterTrack, modelPath

def restoreFromCheckPoint(model, cfg, modelPath):
    if cfg.fromCheckpoint:
        list_of_files = glob.glob("checkpoints/" + modelPath + "*")  # * means all if need specific format then *.csv
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        model = model.load_from_checkpoint(latest_checkpoint)
    return model

def createTrainer(model, cfg, modelPath, datamodule, logdir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        dirpath="checkpoints/" + modelPath,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_last=True,
        mode="min"
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir + modelPath, log_graph=True)
    trainer = pl.Trainer(auto_lr_find=True,
                         max_epochs=cfg.max_epochs,
                         deterministic=False,
                         accelerator="gpu",
                         devices=1,
                         logger=tb_logger,
                         callbacks=[EarlyStopping("val_loss_epoch", patience=80), checkpoint_callback])
    if cfg.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule, min_lr=1e-16)

        # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        model.learning_rate = new_lr
    return model, trainer
def experiment(cfg, datamodule, seq_len, n_features_per_elem):

    modelPath = "BATCHSIZE="+str(cfg.batch_size) + "/"
    logdir = "logs/"
    '''
        1) Get Model
        2) Restore parameters from checkpoint if set
        3) Create a Trainer
        4) Fit 
    '''

    model, parameterTrack, modelPath = getModel(cfg, seq_len, n_features_per_elem, modelPath)
    model = restoreFromCheckPoint(model, cfg, modelPath)
    trainer = createTrainer(model, cfg, modelPath, datamodule, logdir)
    if not cfg.train:
        model.eval()
    else:
        trainer.fit(model, datamodule)

    '''
        5) Predict
    '''
    prediction = trainer.predict(model, dataloaders=datamodule)

    '''
        - Log in wandb
        - Save Loss on file
    '''

    if cfg.onlinelog:
        Benign = [a.item() for (a, b) in prediction if b.item() == 0.0]
        Malign = [a.item() for (a, b) in prediction if b.item() == 1.0]
        Malign += [None]*(len(Benign) - len(Malign))
        data = [[a, b] for a, b in zip(Benign, Malign)]
        table = wandb.Table(data=data , columns=["BenignLoss", "MalignLoss"])
        wandb.log(
            {'LossHistogram': wandb.plot.histogram(table, "Loss", title="Class Loss")}
        )

    from datetime import datetime
    timestr = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    printErrorsDistribution(prediction, logdir + modelPath + "/", "ErrorDistribution_"+timestr+".jpeg", modelName=cfg.model, parameters=parameterTrack)
    with open(logdir + modelPath + "LossPrediction.pkl", "wb") as f:
        pickle.dump(prediction, f)

    '''
        - Additional Tests
    '''
    dns_datamodule = POSTransactionDataLoader(cfg.datasets.pos.dns_normal_data_path,
                                          cfg.datasets.pos.dns_abnormal_data_path,
                                          cfg.batch_size)

    dns_prediction = trainer.predict(model, dataloaders=dns_datamodule)
    with open(logdir + modelPath + "DNSLossPrediction.pkl", "wb") as f:
        pickle.dump(dns_prediction, f)

    https_datamodule = POSTransactionDataLoader(cfg.datasets.pos.https_normal_data_path,
                                              cfg.datasets.pos.https_abnormal_data_path,
                                              cfg.batch_size)

    https_prediction = trainer.predict(model, dataloaders=https_datamodule)
    with open(logdir + modelPath + "HTTPSLossPrediction.pkl", "wb") as f:
        pickle.dump(https_prediction, f)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    if cfg.onlinelog:
        wandb.init(project=cfg.logConfig.project,
                   entity=cfg.logConfig.entity,
                   config=OmegaConf.to_container(cfg))

    datamodule = POSTransactionDataLoader(cfg.datasets.pos.normal_data_path,
                                          cfg.datasets.pos.abnormal_data_path,
                                          cfg.batch_size)
    seq_len = cfg.datasets.pos.seq_len
    n_features_per_elem = cfg.datasets.pos.n_features_per_elem


    experiment(cfg, datamodule, seq_len, n_features_per_elem)


if __name__ == "__main__":

    main()