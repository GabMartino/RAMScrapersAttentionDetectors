import torch
from einops import rearrange
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding
import pytorch_lightning as pl

class Transformer(torch.nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            causal=False,
            dropout=0.2,
            seq_len=None,
            pos_embedding=False,
            cross_attend=False
    ):
        super(Transformer, self).__init__()

        self.attn_layers = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            cross_attend=cross_attend,
            causal=causal
        )

        self.pos_embedding = None
        if pos_embedding:
            assert seq_len is not None, "Must specify seq_len when using positional embeddings"
            self.pos_embedding = AbsolutePositionalEmbedding(dim, seq_len)
        self.norm = torch.nn.LayerNorm(self.attn_layers.dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, return_attention=False, **kwargs):
        if self.pos_embedding is not None:
            embeddings = embeddings + self.pos_embedding(embeddings)

        embeddings = self.dropout(embeddings)
        latent, intermediates = self.attn_layers(embeddings, return_hiddens=True, **kwargs)
        latent = self.norm(latent)

        if return_attention:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return latent, attn_maps

        return latent

class TransformerAE(pl.LightningModule):
    def __init__(self, learning_rate, dim, depth, heads, seq_len, n_features, mask_prob=0.2):
        super(TransformerAE, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.mask_prob = mask_prob
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=False,
            pos_embedding=True,
            seq_len=seq_len
        )

        self.mask = torch.nn.Parameter(torch.randn(dim))
        self.proj_in = torch.nn.Linear(n_features, dim)
        self.proj_out = torch.nn.Linear(dim, n_features)

    def training_step(self, batch, batch_idx):
        x = batch
        predicted_x = self.forward(x)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        loss = loss / batch.shape[0]
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        final_loss = sum(output['loss'] for output in outs) / len(outs)

        self.log('train_loss_epoch', final_loss)

        #wandb.log({"train_loss_epoch": final_loss}) if self.onlinelog else None

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
        #wandb.log({"val_loss_epoch": final_loss}) if self.onlinelog else None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, label = batch
        predicted_x = self.forward(x)
        x = x.reshape(predicted_x.shape)
        loss = torch.nn.L1Loss(reduction='sum')(predicted_x, x)
        return loss, label
    # x = (B, T, N)
    def forward(self, x):
        b, t, n = x.shape
        x_proj = self.proj_in(x)
        mask = torch.rand(b, t) <= self.mask_prob
        x_proj[mask] = self.mask

        hidden = self.transformer(x_proj)

        return self.proj_out(hidden)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)