# %%
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
import torchmetrics as tm

from sofamo.training import get_device
from sofamo.training import get_transforms
from sofamo.training import get_datasets
from sofamo.training import get_model
from sofamo.common.plot_sample import plot_sample
from sofamo.common.onehot import make_onehot
from sofamo.common.to_tensor import to_tensor


# %%
def parse_args(*args, **kwargs):
    return {
        "seed": "42",
        "eps": "0.000001",
        "device": "cuda:0",
        "paths": {
            "project": "/home/sofa/space/prog/diploma/",
            "logs": "/home/sofa/space/prog/diploma/logs/",
            "models": "/home/sofa/space/prog/diploma/models/",
            "data_train": "/home/sofa/space/prog/diploma/data/train_100/",
            "data_valid": "/home/sofa/space/prog/diploma/data/val/",
            "data_test": "/home/sofa/space/prog/diploma/data/test/",
        },
    }


def imshow_sample(image, mask):
    image = image[0, :, :, :].cpu().detach().numpy()
    mask = mask[0, :, :, 1].cpu().detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image, interpolation="nearest")
    ax[1].imshow(mask, interpolation="nearest", cmap="jet")
    plt.show()


def restore_model(fn="./model_full.pt"):
    """Restore full model"""
    model = torch.load(str(fn))
    return model


def restore_params(model, fn="./model_params.pth"):
    """restore model's parameters"""
    state_dict = torch.load(str(fn))
    model.load_state_dict(state_dict)
    return model


def load_image(fname):
    image = cv2.imread(str(fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_stage(image: np.array):
    _, _, test_transforms = get_transforms(params=None)
    image = test_transforms(image=image)["image"]
    image = image.unsqueeze(0)
    image = image.float()
    return image


def process_stage(image: torch.Tensor, model: nn.Module):
    model.train(False)
    with torch.no_grad():
        image = image.to(get_device(model))
        predict = model(image)
    return predict


def postprocess_stage(image: torch.Tensor):
    image = image[0, ...]
    image = nn.functional.softmax(image, dim=0)
    image = image.cpu().detach().numpy()
    return image


def get_device(value):
    if isinstance(value, nn.Module):
        value = next(value.parameters())
    else:
        value = value
    try:
        device = value.device
    except:
        raise ValueError("Wrong type of input")
    return device


def evaluate(image: np.array, model=None):
    if not model:
        raise ValueError("Model must be here")
    image = preprocess_stage(image)
    predict = process_stage(image, model)
    mask = postprocess_stage(predict)
    return mask


def compute_torchmetrics(pred, target, **kwargs):
    metrics = {}
    device = get_device(pred)
    # pred = to_tensor(pred)
    # target = to_tensor(target)
    # init
    fn_confmat = tm.ConfusionMatrix(**kwargs).to(device)
    fn_acc = tm.Accuracy(**kwargs).to(device)
    fn_dice = tm.Dice(**kwargs).to(device)
    fn_jaccard = tm.JaccardIndex(**kwargs).to(device)
    fn_F1 = tm.F1Score(**kwargs).to(device)
    # compute
    metrics["confmat"] = fn_confmat(pred.argmax(0), target).cpu().numpy()
    metrics["accuracy"] = fn_acc(pred.argmax(0), target).cpu().numpy()
    metrics["dice"] = fn_dice(pred.argmax(0), target).cpu().numpy()
    metrics["jaccard"] = fn_jaccard(pred.argmax(0), target).cpu().numpy()
    metrics["f1"] = fn_F1(pred.argmax(0), target).cpu().numpy()
    return metrics


def compute_smp_metrics(pred, target, **kwargs):
    metrics = {}
    # pred = to_tensor(pred)
    # target = to_tensor(target)

    if kwargs["mode"] == "multiclass":
        pred = pred.argmax(0)
        pred = pred.int()
        pred = pred.unsqueeze(0)  # add batch shape
        target = target.int()
        target = target.unsqueeze(0)  # add batch shape
    else:
        raise ValueError("We want mode='multiclass' only.")

    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, **kwargs)
    metrics["TP"] = tp.numpy()
    metrics["FP"] = fp.numpy()
    metrics["FN"] = fn.numpy()
    metrics["TN"] = tn.numpy()
    metrics["accuracy"] = smp.metrics.accuracy(tp, fp, fn, tn).numpy()
    metrics["iou"] = smp.metrics.iou_score(tp, fp, fn, tn).numpy()
    metrics["balanced_accuracy"] = smp.metrics.balanced_accuracy(tp, fp, fn, tn).numpy()
    metrics["f1"] = smp.metrics.f1_score(tp, fp, fn, tn).numpy()
    return metrics


if __name__ == "__main__":
    params = parse_args()

    device = torch.device(params["device"])

    train_transforms, valid_transforms, test_transforms = get_transforms(params)
    train_ds, valid_ds, test_ds = get_datasets(
        train_transforms,
        valid_transforms,
        test_transforms=None,
        params=params,
    )

    model = get_model(params)
    fn_weight = Path(params["paths"]["models"]) / "weights"
    fn_weight = fn_weight / "20230416_203613_e187_gs06000.pth"
    restore_params(model, fn=fn_weight)

    for N in range(0, 10, 10):
        image, mask = test_ds[N]

        mask_pred = evaluate(image=image, model=model)

        # image = to_tensor(image).float()
        image = preprocess_stage(image)
        mask = to_tensor(mask).int()
        mask_pred = to_tensor(mask_pred).float()

        metrics = compute_torchmetrics(
            mask_pred,
            mask,
            task="multiclass",
            num_classes=2,
            ignore_index=None,
            # mode="micro",
            # mdmc_average="global",
        )

        # for key, val in metrics.items():
        #     print(key, val)

        print()
        metrics = compute_smp_metrics(
            mask_pred,
            mask,
            mode="multiclass",
            num_classes=2,
            ignore_index=None,
        )
        for key, val in metrics.items():
            print(key, val)

        # break
        # imshow
        fig, ax = plot_sample(
            image,
            mask.unsqueeze(0),
            mask_pred.unsqueeze(0),
        )
        plt.show()
        # plt.figure(figsize=(10,8))
        # plt.imshow(mask_pred.sum(dim=1)[0,...].cpu().detach(), vmin=0, vmax=1)
        # plt.show()

# %%
