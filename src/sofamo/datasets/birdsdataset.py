# %%
from pathlib import Path
from itertools import product

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

# def _to_onehot(self, codec, values):
#     value_idxs = codec.transform(values)
#     return torch.eye(len(codec.classes_))[value_idxs]


# %%
def rgb(c0, c1, c2, a=None):
    color = (c0, c1, c2) if a is None else (c0, c1, c2, a)
    return color


def load_image(fname):
    image = cv2.imread(str(fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def intersection_filenames(flist0, flist1):
    flist0_pruned = []
    flist1_pruned = []
    for fn0, fn1 in product(flist0, flist1):
        if fn0.stem == fn1.stem:
            flist0_pruned.append(fn0)
            flist1_pruned.append(fn1)
    return list(zip(flist0_pruned, flist1_pruned))

# if __name__ == '__main__':
#     x = ['a.png', 'b.png', 'c.png']
#     y = ['a.png', 'd.png', 'c.png']
#     t = map(lambda i: Path(i), x)
#     s = map(lambda i: Path(i), y)
#     res = intersection_filenames(t, s)
#     print(res)


def to_tensor(image, mask):
    aug = ToTensorV2()
    data_dict = aug(image=image, mask=mask)
    image = data_dict['image']
    mask = data_dict['mask']
    return image, mask


def make_onehot_from_colors(color_mask, colors_list):
    mask = [np.all(color_mask == color, axis=-1) for color in colors_list]
    mask = np.stack(mask, axis=-1)
    mask = mask.astype("float")
    return mask


def make_flat_from_colors(color_mask, colors_list):
    mask = make_onehot_from_colors(color_mask, colors_list)
    mask = mask.argmax(axis=-1)
    mask = mask.astype("float")
    return mask


def represent_mask(mask, colors_list, mask_mode):
    if mask_mode == "one_hot" or mask_mode == 'onehot':
        mask = make_onehot_from_colors(mask, colors_list)
    elif mask_mode == "flat" or mask_mode == "flatten":
        mask = make_flat_from_colors(mask, colors_list)
    return mask


# %%
class BirdsDataset(torch.utils.data.Dataset):
    """The Caltech-UCSD Birds-200-2011 Dataset"""
    DFLT_CLASSES = {
        "background": rgb(0, 0, 0),
        "birds": rgb(255, 255, 255),
    }

    def __init__(
        self,
        filenames: list,
        transforms=None,
        to_tensor=False,
        classes=DFLT_CLASSES,
        mask_mode="flat",
    ):
        self.filenames = filenames
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.mask_mode = mask_mode
        self.classes = classes

    def __getitem__(self, index):
        image = load_image(self.filenames[index][0])
        mask = load_image(self.filenames[index][1])
        mask = np.where(mask < 170, 0, 255).astype('uint8')
        mask = represent_mask(mask, self.classes.values(), self.mask_mode)
        if self.transforms:
            data_dict = self.transforms(image=image, mask=mask)
            image, mask = data_dict["image"], data_dict["mask"]
        if self.to_tensor:
            image, mask = to_tensor(image, mask)
        return image, mask

    def __len__(self):
        return len(self.filenames)


# %% 
if __name__ == "__main__":
    path_root = Path("/home/sofa/space/prog/diploma/data/processed/val/")
    path_img = path_root / 'image'
    path_mask = path_root / 'label'
    fns_images = sorted(list(path_img.glob("**/*.jpg")))
    fns_masks = sorted(list(path_mask.glob("**/*.png")))
    filenames = intersection_filenames(fns_images, fns_masks)

    transforms = A.Compose([
        A.PadIfNeeded(512, 512, always_apply=True),
        A.RandomCrop(512, 512, always_apply=True),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True),
        # ToTensorV2(always_apply=True),
    ])

    train_dataset = BirdsDataset(
        filenames=filenames,
        transforms=transforms,
        to_tensor=False,
        mask_mode='onehot',
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    def imshow_sample(image, mask, i):
        import matplotlib.pyplot as plt

        image = image[i, :, :, :].cpu().detach().numpy()
        mask = mask[i, :, :, 1].cpu().detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image, interpolation="nearest")
        ax[1].imshow(mask, interpolation="nearest", cmap='jet')
        plt.show()

    for i in range(8):

        for step, (image, mask) in enumerate(train_dataloader):
            # image = torch.unsqueeze(image, dim=1)
            # mask = torch.unsqueeze(mask, dim=1)
            imshow_sample(image, mask, i)
            print(image.shape)
            print(mask.shape)
            break

# %%
if __name__ == "__main__":
    path_root = Path('/home/sofa/space/prog/diploma/data/processed/test/')
    name = '004.groove_billed_ani/groove_billed_ani_0044_1731'
    img_path = path_root / 'image' / f'{name}.jpg'
    mask_path = path_root / 'label' / f'{name}.png'
    
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.show()
    mask = plt.imread(mask_path)
    plt.imshow(mask, cmap='binary')
    plt.show()

    # # im = cv2.imread(img_path)
    # # ma = cv2.imread(mask_path) 



# %%
