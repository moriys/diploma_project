# %%
from pathlib import Path
import json
import requests
import io
import os
import sys
import base64
import random

import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image, JpegImagePlugin, PngImagePlugin

from sofamo.common.seed_everything import seed_everything


# %%
def visualize(**images):
    """Plot images in one row"""
    n_images = len(images)
    plt.figure(figsize=(8, 3))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image, interpolation='nearest')
    plt.show()


# %%
def request_anyway(**kwargs):
    try:
        response = requests.post(**kwargs)
    except:
        response = request_anyway(**kwargs)
    return response


# %%
def generate_new_image(image_path, mask_path, params):

    sat_image = Image.open(image_path)
    image_bytes = io.BytesIO()
    sat_image.save(image_bytes, format='PNG')
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    sat_mask = Image.open(mask_path)
    mask_bytes = io.BytesIO()
    sat_mask.save(mask_bytes, format='PNG')
    mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')

    # local web-server Stable Diffusion WebUI
    URL = params['URL']

    prompt = params['stable_diffusion']['prompt']
    random_prompt = params['stable_diffusion']['random_prompt']
    if random_prompt:
        prompt = random.sample(prompt, 1)[0]
    else:
        prompt = prompt
    print(prompt)
    payload = {
        "init_images": [image_base64],
        "mask": mask_base64,
        "denoising_strength": params['stable_diffusion']['denoising_strength'],
        "prompt":  prompt,
        "negative_prompt": params['stable_diffusion']['negative_prompt'],
        "steps": params['stable_diffusion']['steps'],
        "inpainting_mask_invert": params['stable_diffusion']['inpainting_mask_invert'],
        "cfg_scale": params['stable_diffusion']['cfg_scale'],
        "seed": params['stable_diffusion']['seed'],
        "sampler_index": params['stable_diffusion']['sampler_index'],
        "inpainting_fill": params['stable_diffusion']['inpainting_fill'],
        "batch_size": params['stable_diffusion']['batch_size'],
        "mask_blur": 0, # from 0 to 64, int
    }

    response = request_anyway(
        url=f'{URL}/sdapi/v1/img2img',
        json=payload,
        timeout=50,
    )

    r = response.json()

    for i in r['images']:
        gen_image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = request_anyway(
            url=f'{URL}/sdapi/v1/png-info',
            json=png_payload,
        )

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        # gen_image.save('new_background.png', pnginfo=pnginfo)

    visualize(
        original_image=sat_image,
        mask=sat_mask,
        generated_image=gen_image
    )

    return gen_image


# %%
def create_variants(path_in, path_out, params, limit_source_images=None, variants_per_image=1, mode='back'):
    
    path_in = Path(path_in)
    path_in_img = path_in / 'image'
    path_in_mask = path_in / 'label'
    path_out = Path(path_out)
    (path_out / 'image').mkdir(parents=True, exist_ok=True)
    (path_out / 'label').mkdir(parents=True, exist_ok=True)
    

    fns_img = path_in_img.glob("*.jpg")
    fns_img = [fn.name for fn in fns_img if "variant" not in fn.stem]
    fns_mask = path_in_mask.glob("*.png")
    fns_mask = [fn.name for fn in fns_mask if "variant" not in fn.stem]
    fns_img = sorted(fns_img)
    fns_mask = sorted(fns_mask)

    if len(fns_img) == 0 or len(fns_mask) == 0:
        path_checked = fns_img if len(fns_img) == 0 else fns_mask
        raise ValueError("No files found at path: " + str(path_checked))

    path_out.mkdir(parents=True, exist_ok=True)

    counter = 0
    for i, (fn_img, fn_mask) in enumerate(zip(fns_img, fns_mask)):
        if counter >= limit_source_images:
            break
        counter += 1

        print(f"Generating variants for source image {counter}/{limit_source_images}")

        path_img = path_in_img / fn_img
        path_mask = path_in_mask / fn_mask

        for i in range(variants_per_image):

            # Generate and save image variant
            img_variant = generate_new_image(path_img, path_mask, params)
            path_variant = path_out / 'image'/ f"{fn_img.split('.')[0]}-{mode}_variant-{i+1}.jpg"
            img_variant.save(str(path_variant))

            # Save mask in new location to match the folder structure expected
            # by load_data.
            mask_image = PIL.Image.open(path_mask).convert("RGB")  # .resize((512, 512))
            path_variant = path_out / 'label' / f"{fn_mask.split('.')[0]}-{mode}_variant-{i+1}.png"
            mask_image.save(str(path_variant))

            print(f"Image: {i+1:04d} - generated")

# %%
def copy_images(path_in, path_out):
    path_in = Path(path_in)
    path_out = Path(path_out) 
    (path_out / 'image').mkdir(parents=True, exist_ok=True)
    (path_out / 'label').mkdir(parents=True, exist_ok=True)


    path_images = path_in / 'image'
    path_masks = path_in /'label'
    fns_images = sorted(list(path_images.glob("**/*.ipg")))
    fns_masks = sorted(list(path_masks.glob("**/*.png"))) 
    print(len(fns_images))

    for fn_img in fns_images:
        base_name = fn_img.name
        src_image = fn_img
        dest_image = path_out / 'image' / base_name
        dest_image.write_bytes(src_image.read_bytes())

    for fn_lab in fns_masks:
        base_name = fn_lab.name
        src_label = path_masks / base_name
        dest_label = path_out / 'label' / base_name
        dest_label.write_bytes(src_label.read_bytes())

# %%

if __name__ == "__main__":

    # prompt for birds
    use_random_bird_prompt = True
    if use_random_bird_prompt:
        #create promt variants for birds (=our classes in valid and test data)
        path_val = Path('/home/sofa/space/prog/diploma/data/processed/val/image/')
        path_test = Path('/home/sofa/space/prog/diploma/data/processed/test/image/')
    
        prompt_list_birds = []
        for child in path_val.iterdir():
            child_name = ' '.join(child.name.split('.')[-1].split('_'))
            prompt_list_birds.append(child_name)
        for child in path_test.iterdir():
            child_name = ' '.join(child.name.split('.')[-1].split('_'))
            prompt_list_birds.append(child_name)
    else:
        prompt_list_birds = 'one bird, high resolution, 4k'
    

    #prompt variants for background
    prompt_list_background = ['bird in the forest, high resolution, 4k',
                                'bird on the sea, high resolution, 4k',
                                'bird in the city, high resolution, 4k',
                                'bird on the water surface, high resolution, 4k',
                                'bird on the street, high resolution, 4k',
                                'bird in the garden, high resolution, 4k',
                                'bird on the beach, high resolution, 4k',
                                'bird on the road, high resolution, 4k',
                                'bird on the tree, high resolution, 4k',
                                'bird on the table, high resolution, 4k',
                                'bird on the roof, high resolution, 4k']

    negative_prompt_background = "((bird)), ((birds)), bird's wing, bird's tail, bird's beak, ((animals)), deformed, bad anatomy, ugly, disfigured, disgusting, mutated"
    negative_prompt_birds = "deformed, bad anatomy, ugly, disfigured, disgusting, mutated"
    
    bird_mode = 0
    background_mode = 1
    
    # params for Stable Diffusion Inpaint
    params = {
        'URL': "http://10.0.0.3:8765", # local web-server Stable Diffusion
        'stable_diffusion': {
            "denoising_strength": 0.8, #  0.75 for bird, 0.8 for background
            "random_prompt": True,
            "prompt": prompt_list_background,
            "negative_prompt": negative_prompt_background,
            "steps": 25,
            "inpainting_mask_invert": background_mode, # 1 for background, 0 for bird
            "cfg_scale": 7, # 6 for bird, 7 for background  
            "seed": -1,
            "sampler_index": 'LMS', # for bird "Euler a", for background 'LMS'
            "inpainting_fill": 1, 
            "batch_size": 1,
        }
    }
    
    NUM_SAMPLES = 100

    seed_everything(42)

    path_in = Path("/home/sofa/space/prog/diploma/data/processed/train_100/")
    path_out = Path(f"/home/sofa/space/prog/diploma/data/processed/train_row100_gen1000_no_random_class_birds") 
    path_out.mkdir(parents=True, exist_ok=True)

    # create new variants for our images
    create_variants(path_in, path_out, params=params, 
                    limit_source_images=NUM_SAMPLES, 
                    variants_per_image=5, mode='back')
    
    # copy original image and mask to target directory
    #copy_images(path_in, path_out)



# %%
