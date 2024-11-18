# Image Manipulation with Stable Diffusion: Advanced Inpainting Technique

### Tools Used: Python, StableDiffusionInpaintPipeline, Torch

## Aim

<p align="justify">
The primary objective of this project is to develop and implement advanced inpainting techniques to remove and replace specific elements in images. The project encompasses multiple phases: segmentation, inpainting to remove squirrels, and inpainting to replace birds with new bird representations, culminating in a comprehensive set of transformed images.
</p>

## Introduction

<p align="justify">
In this project, We have created a model which detects an object from an image using YOLO, segments the detected object using Facebook's SAM, and in-paints the detected model according to the given prompt using the Stable Diffusion model from Huggingface.
</p>

## Dataset

<p align="justify">
We have used the different birds and squirrels images near a birdfeeder.
</p>

<ul>
<li>Birds</li>
<p align="start">
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/birds/bird-1.jpeg" alt="Bird Image 1" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/birds/bird-2.jpeg" alt="Bird Image 2" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/birds/bird-3.jpeg" alt="Bird Image 3" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/birds/bird-4.jpeg" alt="Bird Image 4" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/birds/bird-5.jpeg" alt="Bird Image 5" width="200"/>
</p>
<li>Squirrels</li>
<p align="start">
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/squirrels/squirrel-1.jpeg" alt="Squirrel Image 1" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/squirrels/squirrel-2.jpeg" alt="Squirrel Image 2" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/squirrels/squirrel-3.jpeg" alt="Squirrel Image 3" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/squirrels/squirrel-4.jpeg" alt="Squirrel Image 4" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/squirrels/squirrel-5.jpeg" alt="Squirrel Image 5" width="200"/>
</p>
</ul>

## Method

## Phase 1: Image Segmentation

<p align="justify">
Segmentation is the process of partitioning an image into multiple segments (sets of pixels) to simplify the representation of an image and make it more meaningful and easier to analyze. In this project, segmentation is crucial for identifying and isolating specific elements, such as squirrels and birds, to be targeted for inpainting. Techniques like YOLO (You Only Look Once) for object detection and SAM (Segment Anything Model) for segmentation are employed to accurately detect and segment the desired elements in the images.
</p>

The segmentation process can be broken down into several key steps:

1. **Object Detection:** Utilizing models like YOLO (You Only Look Once) to detect and localize objects within an image. YOLO is a real-time object detection system that divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell.

2. **Segmentation Mask Generation:** Employing models like SAM (Segment Anything Model) to generate precise segmentation masks for the detected objects. SAM is designed to segment any object in an image, given a prompt such as a point or a box.

**Implementation**

- **Tools Used:** Python, YOLO, SAM, PIL, NumPy, Matplotlib
- **Segmentation of Birds and Squirrels:** The segmentation process is applied to both birds and squirrels. The detected objects are then used to create segmentation masks, which are saved for further processing.

## Phase 2: Object Removal (Removing Squirrels)

<p align="justify">
Inpainting is a technique used to reconstruct missing or damaged portions of an image. In this phase, we employ the Stable Diffusion Inpainting Pipeline to remove squirrels from images. This pipeline leverages advanced algorithms to ensure seamless and natural-looking results by filling the masked regions with appropriate background content.
</p>

<p align="justify">
It leverages surrounding information to seamlessly fill in the gaps, making the image appear complete and natural. In this phase, inpainting is used to remove squirrels from images.
</p>

**Inpainting Techniques:** Traditional inpainting methods involve diffusing information from the boundary of the missing region inward. Modern techniques, such as those based on deep learning, use generative models to fill in the missing regions more intelligently.

**Stable Diffusion Inpainting:** This project utilizes the Stable Diffusion Inpainting Pipeline, a pre-trained model capable of high-resolution inpainting. The model is guided by prompts to generate detailed and realistic images.

**Implementation**

- **Inpainting Pipeline Initialization:** The inpainting pipeline is initialized using StableDiffusionInpaintPipeline, configured with the pre-trained model from "runwayml/stable-diffusion-inpainting" and operated on CPU.
- **Squirrel Removal:** The `remove_squirrels` function processes input image files and corresponding masks, resizing them to (512, 512) for consistency. The inpainting pipeline removes squirrels, and the output images are saved with filenames appended by "-squirrelsRemoved.jpeg."
- **Tools Used:** Python, StableDiffusionInpaintPipeline, Torch, PIL, NumPy, Matplotlib

```python
def initialize_inpaint_pipeline():
    """
    Initialize the inpainting pipeline using StableDiffusionInpaintPipeline.
    Returns:
    inpaint_pipeline (StableDiffusionInpaintPipeline): Initialized inpainting pipeline.
    """
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
    inpaint_pipeline = inpaint_pipeline.to("cpu")
    return inpaint_pipeline
```

# Phase 3: Object Replacement (Replacing Birds)

<p align="justify">
Generative models are used to create new, realistic representations of objects within an image. This phase involves replacing birds with new, distinct bird representations using generative models.
</p>

## Generative Adversarial Networks (GANs)

<p align="justify">
GANs consist of a generator and a discriminator. The generator creates new images, while the discriminator evaluates their authenticity. This adversarial process improves the quality of the generated images over time.
</p>

## Stable Diffusion Models

<p align="justify">
Stable Diffusion models are a type of generative model that uses a diffusion process to generate high-quality images. They are particularly effective for inpainting tasks, where the goal is to fill in missing regions of an image with realistic content.
</p>

## Implementation

### Generative Model

<p align="justify">
The project adopts the Stable Diffusion Inpainting Pipeline, a pre-trained model capable of high-resolution inpainting. Each image is paired with its mask, and a prompt ("Replace bird, high resolution") guides the generative model to create new bird-like elements.
</p>

### Integration with Previous Phases

<p align="justify">
This phase builds upon the inpainting techniques used in the squirrel removal project, introducing generative models to replace birds and integrating both segmented and generated components.
</p>

### Tools Used

<p align="justify">
- Python
- Stable Diffusion Inpainting Pipeline
- Torch
- PIL
- NumPy
- Matplotlib
</p>

```python
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
```


# Dependencies
<ul>
  <li>huggingface</li>
  <li>https://huggingface.co/runwayml/stable-diffusion-inpainting</li>
  <li>https://huggingface.co/hustvl/yolos-tiny</li>
  <li>Pytorch</li>
  <li>tensorflow</li>
  <li>numpy</li>
  <li>matplotlib</li>
</ul>


Results
<p align="justify"> Below image represent the actual image, the masked image and in-painted image. The project yielded a gallery of transformed images where squirrels were seamlessly removed, and birds were replaced with new, visually engaging species. The modified images blend inpainted regions smoothly, with few traces of the original objects. The filenames of the modified images follow a clear convention, allowing for easy identification of the transformed compositions. </p>

Conclusion
<p align="justify"> This comprehensive image manipulation project showcases the elegance and sophistication of modern computer vision techniques. By integrating segmentation, inpainting, and generative modeling, the project transcends conventional image editing, offering a glimpse into the complexity and artistry underlying seemingly simple image transformations. The final set of images demonstrates the successful application of these advanced techniques, resulting in cohesive and visually engaging compositions. </p>


<p align="start">
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/bird-1.jpeg" alt="Actual Image" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/masked_bird-1.jpeg" alt="Masked Image" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/bird-1-replaced.jpeg" alt="Bird Replaced Image" width="200"/>
</p>
<p align="start">
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/squirrel-1.jpeg" alt="Actual Image" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/masked_squirrel-1.jpeg" alt="Masked Image" width="200"/>
  <img src="https://github.com/SanthoshV14/text-to-image-generation/blob/main/images/squirrel-1-removed.jpeg" alt="Bird Replaced Image" width="200"/>
</p>


# Bird Replacement
<section class="feature left">
<div class="image-container">
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/birds1.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/Birds1_Replaced.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/birds2.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="[images/Project4/Birds2_Replaced.jpg](https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/Birds2_Replaced.jpg)" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/birds3.jpg" alt="" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/Birds3_Replaced.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/birds4.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/Birds4_Replaced.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/birds5.jpg" alt="Err!" width="200" height="200"/></a>
  <img src="https://github.com/Urviskumar/Inpainting-Birds-with-Generative-Models/blob/main/Birds5_Replaced.jpg" alt="Err!" width="200" height="200"/></a>
</div>
</section>





# Author
Santhosh Vadivel </br>
Urvishkumar Bharti </br>
Vikram Shahpur </br>
Email - ssansh3@gmail.com </br>
Email - urviskumar.bharti@gmail.com </br>
LinkedIn - https://www.linkedin.com/in/santhosh-vadivel-2141b8126/ </br>
