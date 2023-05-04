# %%
import torch


def save_model(model, fn="./model_full.pt"):
    """ Save Net """
    torch.save(model, str(fn))


def save_params(model, fn="./model_params.pth"):
    """ Save only the parameters """
    torch.save(model.state_dict(), str(fn))

