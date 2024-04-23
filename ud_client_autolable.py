import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from PIL import Image
from io import BytesIO
import base64
import json

filename = "./assets/car.jpeg"
text_prompt = "person"
box_threshold = 0.3
text_threshold = 0.25

image_pil = Image.open(filename).convert("RGB")
with open(filename, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image': image_data,
    'text_prompt': text_prompt,
    'box_threshold': box_threshold,
    'text_threshold': text_threshold
}

response = requests.post('http://localhost:8866/predict', json=data)

# Parse response, retrieve the masks, boxes, phrases, and logits from the result
result = response.json()
masks = np.array(result['masks'])
boxes = np.array(result['boxes'])
logits = np.array(result['logits'])
phrases = result['phrases']

labeled_image = result['labeled_image']
labeled_image = Image.open(BytesIO(base64.b64decode(labeled_image)))

# Display the image with bounding boxes and masks
fig = plt.figure(figsize=(15, 7))
plt.imshow(labeled_image)
plt.title("BBoxes and Segmations")
plt.axis('off')
plt.tight_layout()
plt.show()