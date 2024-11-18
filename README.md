# Image Manipulation with Stable Diffusion: Advanced Inpainting Technique 
### Tools Used: Python, StableDiffusionInpaintPipeline, Torch

# Introduction
In this project, I have created a model which detects an object from an image using YOLO segments the detected object using Facebook's SAM and in-paints the detected model according to the given prompt using stable diffusion model from Huggingface.

# Dataset
I have used the different birds and squirrels images near a birdfeeder.


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

# Results
Below image represent the actual image, the masked image and in-painted image.

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

# Author
Santhos Vadivel </br>
Email - ssansh3@gmail.com </br>
LinkedIn - https://www.linkedin.com/in/santhosh-vadivel-2141b8126/ </br>
