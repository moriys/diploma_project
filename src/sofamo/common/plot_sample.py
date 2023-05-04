# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
# import cv2


# %%
def plot_sample(image, mask, mask_pred, i_sample=None):

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    i_sample = i_sample if i_sample is not None else 0

    image = image[i_sample, ...]
    mask = mask[i_sample, ...]
    mask_pred = mask_pred[i_sample, ...]

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().detach().numpy()

    if len(image.shape) == 3:
        image = np.transpose(image, (1,2,0))
    if len(mask.shape) == 3:
        mask = np.transpose(mask, (1,2,0))
    if len(mask_pred.shape) == 3:
        mask_pred = np.transpose(mask_pred, (1,2,0))
        n_classes = mask_pred.shape[-1]
        mask_pred = mask_pred.argmax(-1)

    image = image * std + mean
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    ax[0].imshow(image)
    ax[1].imshow(mask, interpolation="None", vmin=0, vmax=n_classes-1)
    ax[2].imshow(mask_pred, interpolation="None", vmin=0, vmax=n_classes-1)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    return fig, ax


# %%
if __name__ == "__main__":
    image = np.random.rand(8, 3, 32, 32)  # uniform [0, 1]
    mask = np.random.randint(low=0, high=2, size=(8, 32, 32), dtype=np.uint8)  # uniform [low, high)
    mask_pred = np.random.randint(low=0, high=2, size=(8, 2, 32, 32), dtype=np.uint8)  # uniform [low, high)
    fig, ax = plot_sample(image, mask, mask_pred)
    fig.show()
