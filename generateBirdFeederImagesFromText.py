from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generateBirdFeederImagesFromText():

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    # pipe = pipe.to("cuda")

    prompt = "bird and squirrel fighting near a birdfeeder"
    for i in range(5):
        image = pipe(prompt).images[0]
        
        path = f'./generated-images/generated-image-{i+1}.jpeg'
        path = Path(path)
        if not path.is_file():
            Path('./generated-images').mkdir(parents=True, exist_ok=True)
        plt.imsave(path, np.array(image))

if __name__ == "__main__":
    generateBirdFeederImagesFromText()
    print('generated images are saved in the dir: ./generated-images')