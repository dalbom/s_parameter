from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self, in_feature, out_feature, batchnorm=True, dropout=0.2, activation="gelu"
    ):
        super().__init__()

        self.sequence = nn.Sequential(nn.Linear(in_feature, out_feature))

        if activation == "gelu":
            self.sequence.append(nn.GELU())
        elif activation == "softmax":
            self.sequence.append(nn.Softmax(-1))

        if batchnorm:
            self.sequence.append(nn.BatchNorm1d(out_feature))

        if dropout != 0:
            self.sequence.append(nn.Dropout(p=dropout))

    def forward(self, x):
        return self.sequence(x)


# For given fixed number of measurements, Transformer assigns proper attention
class TransformerClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = MLPBlock(
            cfg.in_features, cfg.d_model, batchnorm=False, dropout=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = MLPBlock(
            cfg.d_model,
            cfg.out_features,
            batchnorm=False,
            dropout=0,
            activation="softmax",
        )

    def forward(self, x):
        # x: (B, 201, 2) -> (B, 201, 256)
        x = self.embedding(x.permute(0, 2, 1))

        # -> (B, 201, 256)
        x = self.encoder(x)

        # -> (B, 256, 201) -> (B, 256)
        x = self.avgpool(x.permute(0, 2, 1)).squeeze()

        # -> (B, 7)
        x = self.head(x)

        return x


def get_model(cfg):
    return TransformerClassifier(cfg)
