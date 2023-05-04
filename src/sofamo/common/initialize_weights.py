# %%
import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(1e-4)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)
            m.bias.data.zero_()
