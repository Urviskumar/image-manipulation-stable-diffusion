import os
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import yolo as yolo
from utils import sam as sam


def segmentImages(paths):
    if len(paths) == 0:
        print("empty input")
        return []

    masks = []
    for path in paths:
        if not os.path.isfile(path):
            print(f'{path} - file not found')
            return
        
        raw_image = Image.open(path).convert("RGB")
        results = yolo(raw_image=raw_image)

        if len(results['labels']) == 0:
            print(f"{path} - No birds detected")
        else:
            input_boxes = [np.ndarray.tolist(results["boxes"].detach().numpy())]

            masks = sam(raw_image=raw_image, input_boxes=input_boxes)

            masked_image = []
            for i, mask in enumerate(masks[0]):
                if not (results['labels'][i].item() >= 85):
                    masked_image = (masked_image == True) | (mask[:, :, :] == True)
            masked_image = masked_image.float()
            masked_image = masked_image.permute(1, 2, 0)
            filename = path.split('/')[-1]
            mask_path = f'./birdmask-images/{filename}'
            mask_path = Path(mask_path)
            if not mask_path.is_file():
                Path('birdmask-images').mkdir(parents=True, exist_ok=True)
            plt.imsave(f'./birdmask-images/masked_{filename}', masked_image.numpy())
            plt.imsave(f'./birdmask-images/{filename}', np.array(raw_image))


if __name__ == "__main__":
    segmentImages(sys.argv[1:])
    print("The masked bird images are saved in the folder ./birdmask-images/")