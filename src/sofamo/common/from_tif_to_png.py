# %%
from PIL import Image
import os
# %%
def convert_from_tif_to_png(source_dir, target_dir):
    # Create the target dir, if it does not exist already
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate through all the files in the source_dir-directory
    for filename in os.listdir(source_dir):
        file = os.path.join(source_dir, filename)
        if filename.endswith('.tif'):
            im = Image.open(file)
            img_name = filename.split('.')[0]
            im.save(target_dir + '/' + img_name + '.png')

# %%
if __name__=='__main__':
    # directory which should be converted
    source_dir = '/home/sofa/space/prog/data/WHU_buildings/val/label'
    # directory where the converted files should be stored
    target_dir = '/home/sofa/space/prog/data/WHU_buildings_PNGs/val/label'
    
    convert_from_tif_to_png(source_dir, target_dir)

   
# %%
