from torch import nn
import torchsummary


class MLPBlock(nn.Module):
    def __init__(self, in_feature, out_feature, batchnorm=True, dropout=0.2):
        super().__init__()

        self.sequence = nn.Sequential(nn.Linear(in_feature, out_feature), nn.GELU())

        if batchnorm:
            self.sequence.append(nn.BatchNorm1d(out_feature))

        if dropout != 0:
            self.sequence.append(nn.Dropout(p=dropout))

    def forward(self, x):
        return self.sequence(x)


class SimpleMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        blocks = []
        in_features = cfg.in_features

        for out_features in cfg.dims[:-1]:
            blocks.append(MLPBlock(in_features, out_features, dropout=cfg.dropout))
            in_features = out_features

        blocks.append(MLPBlock(in_features, cfg.dims[-1], batchnorm=False, dropout=0))

        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Softmax(dim=-1) if cfg.head == "softmax" else nn.Sigmoid()
        # self.summary()

        # self.bypass_initialized = False

    def forward(self, x):
        x = self.blocks(x)

        return self.head(x)

    def summary(self):
        torchsummary.summary(self, (402,))

    def bypass_for_tsne(self, x):
        for module in self.children():
            print(module)


def get_model(cfg):
    return SimpleMLP(cfg)
