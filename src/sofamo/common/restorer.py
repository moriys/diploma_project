# %%
import torch


def restore_model(fn="./model_full.pt"):
    """ Restore full model """
    model = torch.load(str(fn))
    return model


def restore_params(model, fn="./model_params.pth"):
    """ restore model's parameters """
    state_dict = torch.load(str(fn))
    model.load_state_dict(state_dict)
    return model
