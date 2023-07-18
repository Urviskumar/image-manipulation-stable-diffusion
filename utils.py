import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline

def stable_diffusion(prompt, raw_image, mask_image):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
    image = pipeline(prompt=prompt, image=raw_image, mask_image=mask_image).images[0]
    return image


def sam(raw_image, input_boxes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    inputs = sam_processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)
    image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])

    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    
    return masks


def yolo(raw_image):
    yolo_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    yolo_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    inputs = yolo_processor(images=raw_image, return_tensors="pt")
    outputs = yolo_model(**inputs)
    target_sizes = torch.tensor([raw_image.size[::-1]])
    results = yolo_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    return results