import torch.nn as nn


def weights_xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


def fc_weights_reinit(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
