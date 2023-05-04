# %%
from pathlib import Path
import random

from sofamo.common.choise_images import copy_images
from sofamo.common.seed_everything import seed_everything

def copy_directory(path_in, path_list, path_out):
    path_in = Path(path_in)
    path_out = Path(path_out)

    path_list = [path_in / path for path in path_list] 

    for path in path_list:
        file_list = [file for file in path.iterdir()] 
        copy_images(file_list, path_out / path.name)


def split_classes(path_in, path_out, num_train=1, num_val=1, num_test=1):
    path_in = Path(path_in) 
    path_out = Path(path_out)

    path_img = path_in / 'images' 
    path_msk = path_in / 'segmentations' 
    
    full_cls = [child for child in path_img.iterdir()]
    train_cls = random.sample(full_cls, num_train)
    full_cls = [cl for cl in full_cls if cl not in train_cls]
    val_cls = random.sample(full_cls, num_val)
    full_cls = [cl for cl in full_cls if cl not in val_cls]
    test_cls = random.sample(full_cls, num_test)

    train_cls = [cl.name for cl in train_cls]
    val_cls = [cl.name for cl in val_cls]
    test_cls = [cl.name for cl in test_cls]

    copy_directory(path_img, train_cls, path_out / 'train' / 'image')
    copy_directory(path_msk, train_cls, path_out / 'train' / 'label')
    copy_directory(path_img, val_cls, path_out / 'val' / 'image')
    copy_directory(path_msk, val_cls, path_out / 'val' / 'label')
    copy_directory(path_img, test_cls, path_out / 'test' / 'image')
    copy_directory(path_msk, test_cls, path_out / 'test' / 'label')

if __name__ == '__main__':
    
    seed_everything(42)

    path_in = Path('/home/sofa/space/prog/diploma/data/raw_200cls/')
    # привести названия файлов и папок к нижнему регистру
    # for child in path_in.iterdir():
    #     for file in child.iterdir():
    #         file.rename(str(file).lower())
    path_out = Path('/home/sofa/space/prog/diploma/data/processed/')

    split_classes(path_in, path_out, num_train=150, num_val=25, num_test=25)
# %%
