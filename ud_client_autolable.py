import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from PIL import Image
from io import BytesIO
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
masks = np.array(result['masks'])
boxes = np.array(result['boxes'])
logits = np.array(result['logits'])
phrases = result['phrases']

labeled_image = np.array(result['labeled_image'])

# Display the image with bounding boxes and masks
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image_pil)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(labeled_image)
axes[1].set_title("BBoxes and Segmations")
axes[1].axis('off')
plt.tight_layout()
plt.show()