import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn

class LossHistory:
    def __init__(self):
        self.data = []

    def add_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, val, np.nan])

    def add_train_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, val, np.nan, np.nan])

    def add_epoch_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, np.nan, val])

    def save(self):
        columns = [
            'index', 'epoch', 'sample_count',
            'train_loss', 'val_loss', 'epoch_val_loss']
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv('history.csv', index=False)

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, targets)
        probs = torch.sigmoid(outputs)
        loss = (
            targets*self.alpha*((1. - probs)**self.gamma)*bce_loss + (
            (1. - targets)*(probs**self.gamma)*bce_loss))
        return loss.mean()

def get_class_names(df):
    labels = df['primary_label']
    return labels.unique()

def search_layer(module, layer_type, reverse=True):
    if isinstance(module, layer_type):
        return module

    if not hasattr(module, 'children'):
        return None

    children = list(module.children())
    if reverse:
        children = reversed(children)
    # search for the first occurence recursively
    for child in children:
        res = search_layer(child, layer_type)
        if res:
            return res
    return None

def make_test_augmenters(conf):
    audio_aug = None
    image_aug = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    return  audio_aug, image_aug
