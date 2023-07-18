import os
from PIL import Image
import sys
from utils import yolo as yolo


def subSelectBirdImages(paths):
    if len(paths) == 0:
        print("empty input")
        return []
    
    selected_filenames = []
    for path in paths:
        if not os.path.isfile(path):
            print(f'{path} - file not found')
            return
        
        raw_image = Image.open(path).convert("RGB")
        results = yolo(raw_image=raw_image)
        labels = results['labels']
        if (16 in labels) or (85 in labels) or (86 in labels):
            selected_filenames.append(path)
        
        if len(selected_filenames) >= 5:
            break
    return selected_filenames

if __name__ == "__main__":
    selected_filenames = subSelectBirdImages(sys.argv[1:])
    print(f'subselected bird images are {selected_filenames}')