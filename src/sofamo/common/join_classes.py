# %%
from pathlib import Path

from sofamo.common.choise_images import copy_images

def select_classes(path_in):
    path_in = Path(path_in) 
    class_names = []
    for child in path_in.iterdir():
        class_name = child.name.split('.')[-1]
        if '_' in class_name:
            class_name = child.name.split('_')[-1]
        class_names.append(class_name)
    return sorted(set(class_names))

def join_classes(path_in, path_out):
    path_in = Path(path_in) 
    path_out = Path(path_out)

    #cls_names = select_classes(path_in)

    for child in path_in.iterdir():
        file_list = [sub_child for sub_child in child.iterdir()] 
        cls_name = child.name.split('.')[-1]
        if '_' in cls_name:
            cls_name = cls_name.split('_')[-1]
        copy_images(file_list, path_out / cls_name)

# %%
if __name__ == '__main__':
    
    # path_in = Path('/home/sofa/space/prog/diploma/data/raw/')
    # path_out = Path('/home/sofa/space/prog/diploma/data/processed/')
    # join_classes(path_in / 'segmentations', path_out / 'label')

    path = Path('/home/sofa/space/prog/diploma/data/raw/images')
    for child in path.iterdir():
        file_names = sorted(child.glob('*.jpg'))
        print(f'{child.name}: {len(file_names)}')

# %%
