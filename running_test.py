import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.cm as cm

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)

    # Generate a colormap with number_of_objects different colors
    cmap = cm.get_cmap('tab10')
    label_colors = [cmap(i, bytes=True)[:3] for i in range(len(labels))]  # Discard alpha channel
    
    if len(masks) > 0:
        # Use the generated colors for segmentation masks
        image = draw_segmentation_masks(image, masks=masks, colors=label_colors, alpha=alpha)
    
    if len(boxes) > 0:
        # Use the generated colors for bounding boxes
        image = draw_bounding_boxes(image, boxes, colors=label_colors, labels=labels, width=5)
    
    return image.numpy().transpose(1, 2, 0)

model = LangSAM()

image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "person"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

# Draw the image with bounding boxes and masks
image_with_boxes = draw_image(np.array(image_pil), masks, boxes, phrases)

print("Phrases:", phrases)

# Display the image with bounding boxes and masks
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image_pil)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(image_with_boxes)
axes[1].set_title("BBoxes and Segmations")
axes[1].axis('off')
plt.tight_layout()
plt.show()