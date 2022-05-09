#
# adapted from https://www.kaggle.com/code/tattaka/birdclef2022-submission-baseline/notebook
#
import torch
import torch.nn as nn
from torch.nn import functional as F
import timm


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = self.p
        eps = self.eps
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class AttHead(nn.Module):
    def __init__(
            self, in_chans, num_classes, p=0.5, train_period=15.0, infer_period=5.0):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512, out_channels=num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.fix_scale = nn.Conv1d(
            in_channels=512, out_channels=num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)

    def forward(self, x):
        x = self.pooling(x).squeeze(-2).permute(0, 2, 1)
        x = self.dense_layers(x).permute(0, 2, 1)
        time_att = torch.tanh(self.attention(x))
        assert self.train_period == self.infer_period
        logits = torch.sum(
            self.fix_scale(x) * torch.softmax(time_att, dim=-1), dim=-1)
        return logits


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            conf.arch, conf.pretrained, num_classes=num_classes,
            drop_rate=conf.dropout_rate, features_only=True)
        encoder_channels = self.backbone.feature_info.channels()
        self.head = AttHead(
            encoder_channels[-1], num_classes, p=0.5,
            train_period=conf.duration, infer_period=conf.duration)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x[-1])
