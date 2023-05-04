# %%
import os
import gc
import random
import datetime as dt
from pathlib import Path
from itertools import product
from copy import deepcopy
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose
import segmentation_models_pytorch as smp
import torchmetrics as tm
import torchsummary as ts
import lightning as L

from sofamo.datasets.birdsdataset import BirdsDataset, intersection_filenames
from sofamo.models.unet_v1 import UNet
from sofamo.losses.losses import (
    CE_DiceLoss,
    JaccardLoss,
    DiceLoss,
    TverskyLoss,
    LovaszSoftmax,
)
from sofamo.common.plot_sample import plot_sample
from sofamo.common.initialize_weights import initialize_weights

from torch.utils.tensorboard import SummaryWriter


# %%
print(torch.cuda.is_available())


# %%
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args(*args, **kwargs):
    return {
        "seed": "42",
        "eps": "0.000001",
        "device": "cuda:0",
        "dataset_name": "birds",
        "paths": {
            # "project": "/workspaces/sofa-diploma/",
            # "logs": "/workspaces/sofa-diploma/logs/",
            # "models": "/workspaces/sofa-diploma/models/",
            # "data_train": "/data/data.sets/free-mlspace/whu_dataset/train_full/",
            # "data_valid": "/data/data.sets/free-mlspace/whu_dataset/val/",
            # "data_test": "/data/data.sets/free-mlspace/whu_dataset/test/",

            # "project": "/home/sofa/space/prog/diploma/",
            # "logs": "/home/sofa/space/prog/diploma/logs/",
            # "models": "/home/sofa/space/prog/diploma/models/",
            # "data_train": "/home/sofa/space/prog/diploma/data/train_row_100_gen_100/",
            # "data_valid": "/home/sofa/space/prog/diploma/data/val/",
            # "data_test": "/home/sofa/space/prog/diploma/data/test/",
            
            "project": "/home/sofa/space/prog/diploma/",
            "logs": "/home/sofa/space/prog/diploma/logs/",
            "models": "/home/sofa/space/prog/diploma/models/",
            "data_train": "/home/sofa/space/prog/diploma/data/processed/train_row100_gen1000_no_random_class_birds/",
            "data_valid": "/home/sofa/space/prog/diploma/data/processed/val/",
            "data_test": "/home/sofa/space/prog/diploma/data/processed/test/",
        },
        "train": {
            "batch_size": 25,
        },
        "valid": {
            "batch_size": 10,
        },
        "test": {
            "batch_size": 1,
        },
    }


def get_device(params):
    try:
        device = torch.device(params["device"])
    except RuntimeError as exc:
        device = torch.device("cpu")
    return device


def get_transforms(params=None):
    train = A.Compose(
        [
            A.PadIfNeeded(512, 512, always_apply=True),
            A.RandomCrop(512, 512, always_apply=True),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    valid = A.Compose(
        [
            A.PadIfNeeded(512, 512, always_apply=True),
            A.RandomCrop(512, 512, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    test = deepcopy(valid)
    return train, valid, test


def get_datasets(train_transforms, valid_transforms, test_transforms, params=None):
    path_train = Path(params["paths"]["data_train"])
    path_valid = Path(params["paths"]["data_valid"])
    path_test = Path(params["paths"]["data_test"])

    path_train_img = path_train / 'image'
    path_train_mask = path_train / 'label'
    fns_images = sorted(list(path_train_img.glob("**/*.jpg")))
    fns_masks = sorted(list(path_train_mask.glob("**/*.png")))
    train_filenames = intersection_filenames(fns_images, fns_masks)

    path_valid_img = path_valid / 'image'
    path_valid_mask = path_valid / 'label'
    fns_images = sorted(list(path_valid_img.glob("**/*.jpg")))
    fns_masks = sorted(list(path_valid_mask.glob("**/*.png")))
    valid_filenames = intersection_filenames(fns_images, fns_masks)

    path_test_img = path_test / 'image'
    path_test_mask = path_test / 'label'
    fns_images = sorted(list(path_test_img.glob("**/*.jpg")))
    fns_masks = sorted(list(path_test_mask.glob("**/*.png")))
    test_filenames = intersection_filenames(fns_images, fns_masks)

    train_dataset = BirdsDataset(
        filenames=train_filenames,
        transforms=train_transforms,
        to_tensor=False,
        mask_mode="flatten",
    )
    valid_dataset = BirdsDataset(
        filenames=valid_filenames,
        transforms=valid_transforms,
        to_tensor=False,
        mask_mode="flatten",
    )
    test_dataset = BirdsDataset(
        filenames=test_filenames,
        transforms=test_transforms,
        to_tensor=False,
        mask_mode="flatten",
    )
    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(train_ds=None, valid_ds=None, test_ds=None, params=None):
    train_dl = None
    valid_dl = None
    test_dl = None
    if train_ds is not None:
        train_dl = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=params['train']['batch_size'],
            shuffle=True,
            # pin_memory=True,
            num_workers=4,
            drop_last=True,
        )
    if valid_ds is not None:
        valid_dl = torch.utils.data.DataLoader(
            dataset=valid_ds,
            batch_size=params['valid']['batch_size'],
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
    if test_ds is not None:
        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=params['test']['batch_size'],
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
    return train_dl, valid_dl, test_dl


def get_model(params):
    device = get_device(params)
    # model = UNet(
    #     in_channels=3,
    #     out_channels=2,
    #     # deep_channels=[160, 256, 256, 256, 256],
    #     deep_channels=[64, 128, 256, 512, 1024],
    #     activation="relu",
    #     use_bn=False,
    #     use_assp=True,
    #     is_features_return=False,
    # )
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, #"imagenet",
        # encoder_depth=5,
        classes=2,
        activation=None,
    )
    initialize_weights(model) # if after pretrained model-the weights will have new initialization
    model = model.to(device=device)

    # compiled_model = model
    # try:
    #     import torch._dynamo as dynamo

    #     torch._dynamo.config.verbose = True
    #     torch._dynamo.config.suppress_errors = True
    #     # torch.backends.cudnn.benchmark = True
    #     compiled_model = torch.compile(
    #         model,
    #         mode="max-autotune",
    #         fullgraph=False,
    #     )
    #     print("Model compiled set")
    # except Exception as err:
    #     print(f"Model compile not supported: {err}")

    return model


def get_loss_fn(params):
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = DiceLoss(smooth=float(params["eps"]))
    # loss_fn = JaccardLoss(smooth=float(params["eps"]))
    # loss_fn = CE_DiceLoss(smooth=float(params["eps"]))
    # loss_fn = TverskyLoss(alpha=0.01, beta=0.5)
    # loss_fn = TverskyLoss(alpha=0.1, beta=0.9)
    # loss_fn = TverskyLoss(alpha=0.9, beta=0.1)
    # loss_fn = LovaszSoftmax()
    # loss_fn = smp.losses.DiceLoss(
    #     mode="multiclass",
    #     classes=2,
    #     log_loss=False,
    #     from_logits=True,
    #     smooth=0.0,
    #     eps=1e-07,
    #     ignore_index=None,
    # )
    loss_fn0 = smp.losses.TverskyLoss(
        alpha=0.01,
        beta=0.99,
        gamma=1.0,
        mode="multiclass",
        classes=2,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        eps=1e-07,
        ignore_index=None,
    )
    loss_fn1 = smp.losses.LovaszLoss(
        mode="multiclass",
        per_image=False,
        from_logits=True,
        ignore_index=None,
    )
    def TverskyAndLovaszLoss(pred, target):
        return loss_fn0(pred, target) + loss_fn1(pred, target)
    loss_fn = TverskyAndLovaszLoss
    return loss_fn


class EarlyStopper:
    def __init__(self, mode='min', patience=1, min_delta=0):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        if self.mode == 'min':
            self.metric_value = np.inf
        elif self.mode == 'max':
            self.metric_value = np.array(0.0)

    def step(self, current_metric_value):
        is_break = False
        if self.mode == 'min':
            val0 = current_metric_value
            val1 = self.metric_value
        elif self.mode == 'max':
            val1 = current_metric_value
            val0 = self.metric_value

        if val0 < val1:
            self.metric_value = current_metric_value
            self.counter = 0
        elif val0 > (val1 + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                is_break = True
        return is_break


def get_optimizer(params):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        # params=[
        #     {"params": model.encoder.parameters(), "lr": 1e-5},
        #     {"params": model.decoder.parameters(), "lr": 4e-3},
        #     {"params": model.segmentation_head.parameters()},
        # ],
        lr=4e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=50,
    )
    # scheduler = ChainedScheduler([scheduler0, scheduler1])
    return optimizer, scheduler


def compute_smp_metrics(pred, target, **kwargs):
    metrics = {}
    if kwargs["mode"] == "multiclass":
        pred = pred.argmax(1)
        pred = pred.int()
        target = target.int()
        # pred = pred.unsqueeze(0)  # add batch shape
        # target = target.unsqueeze(0)  # add batch shape
    else:
        raise ValueError("We want mode='multiclass' only.")
    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, **kwargs)
    metrics["TP"] = tp.numpy()
    metrics["FP"] = fp.numpy()
    metrics["FN"] = fn.numpy()
    metrics["TN"] = tn.numpy()
    metrics["accuracy"] = smp.metrics.accuracy(tp, fp, fn, tn).numpy()
    metrics["iou"] = smp.metrics.iou_score(tp, fp, fn, tn).numpy()
    # metrics["balanced_accuracy"] = smp.metrics.balanced_accuracy(
    #     tp, fp, fn, tn).numpy()
    metrics["F1"] = smp.metrics.f1_score(tp, fp, fn, tn).numpy()
    return metrics


if __name__ == "__main__":
    params = parse_args()

    torch.set_float32_matmul_precision('medium')
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto",
        strategy="dp",
    )
    fabric.seed_everything(int(params["seed"]))
    fabric.launch()

    # seed_everything(int(params["seed"]))
    # device = get_device(params)
    date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    path_project = Path(params["paths"]["project"])
    path_data_train = Path(params["paths"]["data_train"])
    path_tb = Path(params["paths"]["logs"]) / f"tensorboard"
    path_tb = path_tb / f"{path_data_train.name}/{date_str}/"
    path_saving = path_project / "models"

    tb_writer = SummaryWriter(log_dir=path_tb, flush_secs=60)

    train_transforms, valid_transforms, test_transforms = get_transforms(
        params,
    )
    train_ds, valid_ds, test_ds = get_datasets(
        train_transforms,
        valid_transforms,
        test_transforms,
        params=params,
    )
    train_dl, valid_dl, test_dl = get_dataloaders(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        params=params,
    )
    train_dl = fabric.setup_dataloaders(train_dl)
    valid_dl = fabric.setup_dataloaders(valid_dl)
    test_dl = fabric.setup_dataloaders(test_dl)

    model = get_model(params)
    optimizer, scheduler = get_optimizer(params)
    model, optimizer = fabric.setup(model, optimizer)

    loss_fn = get_loss_fn(params)
    early_stopper = EarlyStopper(
        mode='max',
        patience=100,
        min_delta=0,
    )

    # train loop
    gs = global_step = 0
    gs_valid = 0
    gs_test = 0
    gh = global_history = {
        "train": defaultdict(list),
        "valid": defaultdict(list),
        "test": defaultdict(list),
    }
    iou_score_valid_best = 0.0

    for epoch in range(1000):
        history = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }
        # ----------------- TRAINING STAGE -----------------------
        stage = "train"
        model.train(True)
        for step, (image, mask) in enumerate(train_dl):
            # image = image.to(device)
            # mask = mask.to(device)
            image = image.float()
            mask = mask.long()
            optimizer.zero_grad()  # clear gradients for this training step
            output = model(image)
            loss = loss_fn(output, mask)
            mask_pred = nn.functional.softmax(output, dim=1)
            # loss.backward()  # backpropagation, compute gradients
            fabric.backward(loss)
            optimizer.step()  # apply gradients
            history[stage]['loss'].append(loss.detach().cpu().numpy())
            history[stage]['epoch'].append(epoch)
            history[stage]['global_step'].append(global_step)

            # metrics computing
            metrics = compute_smp_metrics(
                mask_pred,
                mask,
                mode="multiclass",
                num_classes=2,
                ignore_index=None,
            )
            # metrics append to history
            for key, value in metrics.items():
                history[stage][key].append(value)

            lr = [param_group['lr'] for param_group in optimizer.param_groups]
            history[stage]['lr'].append(lr)

            # printing on batch
            if global_step % 100 == 0:
                loss_value = history[stage]['loss'][-1]
                iou_score = history[stage]['iou'][-1]  # last in history
                iou_score = np.mean(iou_score[:, 1])  # building class only == 1
                lr = history[stage]['lr'][-1]
                print(
                    f"Stage: {stage} | "
                    f"epoch: {epoch:02d} | "
                    f"gs: {global_step:06d} | "
                    f"loss: {loss_value:.5f} | "
                    f"IoU: {iou_score:.5f} | "
                    f"lr: {lr}"
                )
            # loss to tensorboard on batch
            if global_step % 1 == 0:
                loss_value = history[stage]['loss'][-1]
                tb_writer.add_scalar(
                    tag=f"{stage}/loss_value",
                    scalar_value=loss_value,
                    global_step=global_step,
                )
            # LR to tensorboard on batch
            if global_step % 1 == 0:
                learning_rates = history[stage]['lr'][-1]
                for i, lr in enumerate(learning_rates):
                    tb_writer.add_scalar(
                        tag=f"{stage}/lr_{i}",
                        scalar_value=lr,
                        global_step=global_step,
                    )
            # metrics to tensorboard on batch
            if global_step % 1 == 0:
                for key in metrics.keys():
                    value = history[stage][key][-1]
                    for i, class_name in enumerate(train_ds.classes.keys()):
                        tb_writer.add_scalar(
                            tag=f"{stage}/{class_name}/{key}",
                            scalar_value=np.mean(value[:, i]),
                            global_step=global_step,
                        )
            # checkpoint saving
            if global_step % 1000 == 0:
                fn = path_saving / "checkpoints"
                fn = fn / f"{path_data_train.name}_{date_str}_e{epoch:04d}_gs{global_step:05d}.ckpt"
                torch.save(model, str(fn))
            # weights saving
            if global_step % 500 == 0:
                fn = path_saving / "weights"
                fn = fn / f"{path_data_train.name}_{date_str}_e{epoch:04d}_gs{global_step:05d}.pth"
                torch.save(model.state_dict(), str(fn))
            # image saving
            if global_step % 100 == 0:
                fig, ax = plot_sample(
                    image=image,
                    mask=mask,
                    mask_pred=mask_pred,
                )
                tb_writer.add_figure(
                    tag=f"{stage}/sample_example",
                    figure=fig,
                    global_step=global_step,
                    close=True,
                )
                plt.close(fig)
            global_step += 1

        # upgrade global history
        if epoch % 1 == 0:
            global_history[stage]['epoch'].append(epoch)
            global_history[stage]['global_step'].append(global_step)
            loss_avg = np.mean(history[stage]['loss'])
            global_history[stage]['loss'].append(loss_avg)
            for key in metrics.keys():
                score = np.concatenate(history[stage][key], axis=0)
                score_avg = np.mean(score, axis=0)
                global_history[stage][key].append(list(score_avg))

        # printing on epoch
        if epoch % 1 == 0:
            print(
                f"Stage: {stage}_full | "
                f"epoch: {epoch:02d} | "
                f"gs: {global_step:06d} | "
                f"loss: {global_history[stage]['loss'][-1]:.5f} | "
                f"IoU: {global_history[stage]['iou'][-1][1]:.5f} "  # [1] - buildings
            )

        # metrics to tensorboard on epoch
        if epoch % 1 == 0:
            for key in metrics.keys():
                for i, class_name in enumerate(train_ds.classes.keys()):
                    tb_writer.add_scalar(
                        tag=f"{stage}_epoch/{class_name}/{key}",
                        scalar_value=global_history[stage][key][-1][1],  # [1] - buildings
                        global_step=epoch,
                    )
        
        gc.collect(generation=2)

        # ----------------- VALIDATION STAGE -----------------------
        stage = "valid"
        model.train(False)
        with torch.no_grad():
            for step, (image, mask) in enumerate(valid_dl):
                # image = image.to(device)
                # mask = mask.to(device)
                image = image.float()
                mask = mask.long()
                output = model(image)
                loss = loss_fn(output, mask)
                mask_pred = nn.functional.softmax(output, dim=1)
                loss_value = loss.detach().cpu().numpy()
                history[stage]['loss'].append(loss_value)
                history[stage]['epoch'].append(epoch)
                history[stage]['global_step'].append(global_step)

                # metrics computing
                metrics = compute_smp_metrics(
                    mask_pred,
                    mask,
                    mode="multiclass",
                    num_classes=2,
                    ignore_index=None,
                )
                # metrics to history on batch
                for key, value in metrics.items():
                    history[stage][key].append(value)

                # printing on batch
                if step % 100 == 0:
                    iou_score = history[stage]['iou'][-1]  # last in history
                    iou_score = np.mean(iou_score[:, 1])  # building class == 1
                    print(
                        f"Stage: {stage} | "
                        f"epoch: {epoch:02d} | "
                        f"gs: {global_step:06d} | "
                        f"loss: {loss_value:.5f} | "
                        f"IoU: {iou_score:.5f} "
                    )
                # loss to tensorboard on batch
                if step % 1 == 0:
                    loss_value = history[stage]['loss'][-1]
                    tb_writer.add_scalar(
                        tag=f"{stage}/loss_value",
                        scalar_value=loss_value,
                        global_step=gs_valid,
                    )
                # plotting on batch
                if step % 20 == 0:
                    fig, ax = plot_sample(
                        image=image,
                        mask=mask,
                        mask_pred=mask_pred,
                    )
                    tb_writer.add_figure(
                        tag=f"{stage}/sample_example={step}",
                        figure=fig,
                        global_step=epoch,
                        close=True,
                    )
                    plt.close(fig)
                gs_valid += 1

            # upgrade global history
            if epoch % 1 == 0:
                global_history[stage]['epoch'].append(epoch)
                global_history[stage]['global_step'].append(global_step)
                loss_avg = np.mean(history[stage]['loss'])
                global_history[stage]['loss'].append(loss_avg)
                for key in metrics.keys():
                    score = np.concatenate(history[stage][key], axis=0)
                    score_avg = np.mean(score, axis=0)
                    global_history[stage][key].append(list(score_avg))

            # printing on epoch
            if epoch % 1 == 0:
                print(
                    f"Stage: {stage}_full | "
                    f"epoch: {epoch:02d} | "
                    f"gs: {global_step:06d} | "
                    f"loss: {global_history[stage]['loss'][-1]:.5f} | "
                    f"IoU: {global_history[stage]['iou'][-1][1]:.5f} "  # [1] - buildings
                )

            # metrics to tensorboard on epoch
            if epoch % 1 == 0:
                for key in metrics.keys():
                    for i, class_name in enumerate(train_ds.classes.keys()):
                        tb_writer.add_scalar(
                            tag=f"{stage}_epoch/{class_name}/{key}",
                            scalar_value=global_history[stage][key][-1][1],  # [1] - buildings
                            global_step=epoch,
                        )
            
            # save best_model on the best valid_loss on epoch
            if epoch % 1 == 0:
                iou_score = global_history[stage]['iou'][-1][1]
                if iou_score >= iou_score_valid_best:
                    iou_score_valid_best = iou_score
                    fn_ch = path_saving / "checkpoints" / f"best_{path_data_train.name}_{date_str}_iou={iou_score}.ckpt"
                    fn_we = path_saving / "weights" / f"best_{path_data_train.name}_{date_str}_iou={iou_score}.pth"
                    torch.save(model, str(fn_ch))
                    torch.save(model.state_dict(), str(fn_we))

        # step with scheduler ReduceLROnPlateau
        iou_score = global_history[stage]['iou'][-1][1]   # [1] - buildings
        scheduler.step(iou_score)

        # step with early_stopping
        early_status = early_stopper.step(iou_score)
        if early_status is True:
            print("----- EARLY STOPPING -----")
            break

        gc.collect(generation=2)

    # ----------------- TESTING STAGE -----------------------
    stage = "test"
    # history[stage] = defaultdict(list)
    # model = get_model(params)
    # state_dict = torch.load(f='/home/sofa/space/prog/diploma/models/weights/20230429_001837_e0354_gs67000.pth',
    #                         map_location=fabric.device)
    # model.load_state_dict(state_dict)
    model.train(False)
    with torch.no_grad():
        for step, (image, mask) in enumerate(test_dl):
            # image = image.to(device)
            # mask = mask.to(device)
            image = image.float()
            mask = mask.long()
            output = model(image)
            loss = loss_fn(output, mask)
            mask_pred = nn.functional.softmax(output, dim=1)
            loss_value = loss.detach().cpu().numpy()
            history[stage]['loss'].append(loss_value)
            history[stage]['epoch'].append(epoch)
            history[stage]['global_step'].append(global_step)

            # metrics computing
            metrics = compute_smp_metrics(
                mask_pred,
                mask,
                mode="multiclass",
                num_classes=2,
                ignore_index=None,
            )
            # metrics to history on batch
            for key, value in metrics.items():
                history[stage][key].append(value)

            # printing on batch
            if step % 20 == 0:
                iou_score = history[stage]['iou'][-1]  # last in history
                iou_score = np.mean(iou_score[:, 1])  # building class == 1
                print(
                    f"Stage: {stage} | "
                    f"epoch: {epoch:02d} | "
                    f"gs: {global_step:06d} | "
                    f"loss: {loss_value:.5f} | "
                    f"IoU: {iou_score:.5f} "
                )
            # loss to tensorboard on batch
            if step % 1 == 0:
                loss_value = history[stage]['loss'][-1]
                tb_writer.add_scalar(
                    tag=f"{stage}/loss_value",
                    scalar_value=loss_value,
                    global_step=gs_test,
                )
            # plotting on batch
            if step % 20 == 0:
                fig, ax = plot_sample(
                    image=image,
                    mask=mask,
                    mask_pred=mask_pred,
                )
                tb_writer.add_figure(
                    tag=f"{stage}/sample_example",
                    figure=fig,
                    global_step=step,
                    close=True,
                )
                plt.close(fig)
            gs_test += 1

        # upgrade global history
        if epoch % 1 == 0:
            global_history[stage]['epoch'].append(epoch)
            global_history[stage]['global_step'].append(global_step)
            loss_avg = np.mean(history[stage]['loss'])
            global_history[stage]['loss'].append(loss_avg)
            for key in metrics.keys():
                score = np.concatenate(history[stage][key], axis=0)
                score_avg = np.mean(score, axis=0)
                global_history[stage][key].append(list(score_avg))

        # printing on epoch
        if epoch % 1 == 0:
            print(
                f"Stage: {stage}_full | "
                f"epoch: {epoch:02d} | "
                f"gs: {global_step:06d} | "
                f"loss: {global_history[stage]['loss'][-1]:.5f} | "
                f"IoU: {global_history[stage]['iou'][-1][1]:.5f} "  # [1] - buildings
            )

        # metrics to tensorboard on epoch
        if epoch % 1 == 0:
            for key in metrics.keys():
                for i, class_name in enumerate(train_ds.classes.keys()):
                    tb_writer.add_scalar(
                        tag=f"{stage}_epoch/{class_name}/{key}",
                        scalar_value=global_history[stage][key][-1][i],
                        global_step=epoch,
                    )
            

# %%
