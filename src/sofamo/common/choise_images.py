# %%
from pathlib import Path
import random
import shutil

def choise_files(path_in, num=None, use_random=False, suffix='.png', seed=42):
    random.seed(seed)
    path_in = Path(path_in) 

    file_list = sorted(list(path_in.glob(f"**/*{suffix}")))

    if use_random:
        file_list = random.sample(file_list, num)
    else:
        file_list = file_list[:num] if num is not None else file_list
    
    return file_list

def copy_images(fns_in, path_out):
    fns_in = [Path(fn) for fn in fns_in]
    path_out = Path(path_out) 
    path_out.mkdir(parents=True, exist_ok=True)

    for fn_img in fns_in:
        src_name = fn_img
        dest_name = path_out / fn_img.name
        dest_name.write_bytes(src_name.read_bytes())


if __name__ == '__main__':
    
    # create a limited dataset
    quantity = 100
    path_in = Path('/home/sofa/space/prog/diploma/data/processed/train/')
    path_out = path_in.parent / f'train_{quantity}'

    fns_images = choise_files((path_in / 'image'), num=quantity, use_random=True, suffix='.jpg')

    #use for buildings
    #fns_labels = [path_in / 'label' / fn.name for fn in fns_images] 

    # use for birds
    fns_labels = [] 
    for fn_img in fns_images: 
        path = path_in / 'label'
        fn = list(path.glob(f"**/*{fn_img.stem}*"))
        fns_labels.extend(fn)

    copy_images(fns_images, path_out / 'image')
    copy_images(fns_labels, path_out / 'label')
    
# %%
