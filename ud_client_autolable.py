import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.cm as cm
import base64
import json

filename = "./assets/elevator.jpg"
text_prompt = "each button"
box_threshold = 0.3
text_threshold = 0.25
image_pil = Image.open(filename).convert("RGB")

# Open image file in binary mode, convert to base64
with open(filename, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Define the data to send
data = {
    'image': image_data,
    'text_prompt': text_prompt,
    'box_threshold': box_threshold,
    'text_threshold': text_threshold
}

# Send POST request to server
response = requests.post('http://localhost:8866/predict', json=data)

# Parse response
result = response.json()

# retrieve the masks, boxes, phrases, and logits from the result
masks = torch.from_numpy(np.array(result['masks']))
boxes = torch.from_numpy(np.array(result['boxes']))
logits = np.array(result['logits'])
phrases = result['phrases']

# Print phrases
print(phrases)

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

# Draw the image with bounding boxes and masks
image_with_boxes = draw_image(np.array(image_pil), masks, boxes, phrases)

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