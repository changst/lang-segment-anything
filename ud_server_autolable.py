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
from flask import Flask, request, jsonify
import base64
import json

app = Flask(__name__)
model = LangSAM()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert base64 image to PIL image
    image_data = base64.b64decode(data['image'])
    image_pil = Image.open(BytesIO(image_data))

    # Get text prompt from request data
    text_prompt = data['text_prompt']

    # Get box_threshold and text_threshold from request data
    box_threshold = data.get('box_threshold', 0.3)  # Use default value if not provided
    text_threshold = data.get('text_threshold', 0.25)  # Use default value if not provided

    # Call the model's predict method
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)

    # protect against empty results
    if len(boxes) == 0:
        result = {
            'masks': [],
            'boxes': [],
            'phrases': [],
            'logits': []
        }
        return jsonify(result)
    
    # Compresse the masks to save space
    masks = np.packbits(masks, axis=2)

    # Convert results to JSON serializable format
    result = {
        'masks': masks.tolist(),
        'boxes': boxes.tolist(),
        'phrases': phrases,
        'logits': logits.tolist()
    }

    return jsonify(result)

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8866)