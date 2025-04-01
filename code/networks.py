import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, output_layer=False):
        super().__init__()

        self.output_layer = output_layer

        self.linear = nn.Linear(in_dim, out_dim)
        self.bnorm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.linear(x)

        if not self.output_layer:
            x = self.bnorm(x)
            x = self.relu(x)

        return x
    

class Encoder(nn.Module):
    def __init__(self, enc_in_dim, enc_dim, enc_out_dim, proj_dim, proj_out_dim, dropout=None):
        super().__init__()

        layers = [
            LinearLayer(enc_in_dim, enc_dim),
            LinearLayer(enc_dim, enc_dim),
            LinearLayer(enc_dim, enc_dim),
            LinearLayer(enc_dim, enc_out_dim, output_layer=True)
        ]
        if dropout is not None:
            layers.insert(0, nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        self.proj_head = nn.Sequential(
            LinearLayer(enc_out_dim, proj_dim),
            LinearLayer(proj_dim, proj_out_dim, output_layer=True)
        )


    def forward(self, x):
        hiddens = self.encoder(x)
        proj = self.proj_head(hiddens)

        return hiddens, proj