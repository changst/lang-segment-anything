import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from PIL import Image
from io import BytesIO
import base64
import json
import os

def auto_label(filename, text_prompt, box_threshold=0.3, text_threshold=0.25):
    with open(filename, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    data = {
        'image': image_data,
        'text_prompt': text_prompt,
        'box_threshold': box_threshold,
        'text_threshold': text_threshold
    }

    response = requests.post('http://192.168.10.48:8866/predict', json=data)
    # Parse response, retrieve the masks, boxes, phrases, and logits from the result
    label = response.json()
    return label

def save_label(label, filename):
    with open(filename, 'w') as f:
        json.dump(label, f)

def display_labeled_image(image, label, filename=None):
    boxes = np.array(label['boxes'])
    masks = np.array(label['masks'], dtype=np.uint8)
    if masks is not None and masks.ndim == 3:
        masks = np.unpackbits(masks, axis=2)

    logits = np.array(label['logits'])
    phrases = label['phrases']
    # Display the image with bounding boxes and masks
    fig = plt.figure()
    plt.imshow(image)
    plt.title("BBoxes and Segmations")
    plt.axis('off')

    # Display the bounding boxes
    colors = cm.rainbow(np.linspace(0, 1, len(boxes)))
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, lw=1)

        # Display the mask
        mask = masks[i]
        mask = np.ma.masked_where(mask == 0, mask)
        plt.imshow(mask, cmap='cool', alpha=0.5)

        # Display the phrase
        # phrase = phrases[i]
        # fontsize = min(max(min(x2 - x1, y2 - y1) // 10, 3), 10)
        # plt.text(x1, y1, phrase, color='white', fontsize=fontsize)

        # # Display the logits
        # logit = logits[i]
        # plt.text(x1, y2, f"logit: {logit:.2f}", color='white', fontsize=12, backgroundcolor='black')

    plt.tight_layout()
    if filename:
        # save the image with high quality
        plt.savefig(filename, dpi=600)
        plt.close(fig)
    else:
        plt.show()

text_prompt = "each button"
data_directory = "./dataset/"
image_directory = data_directory + "images/"
label_directory = data_directory + "labels/"
labeled_image_directory = data_directory + "labeled_images/"
if not os.path.exists(label_directory):
    os.makedirs(label_directory)
if not os.path.exists(labeled_image_directory):
    os.makedirs(labeled_image_directory)

# iterate all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        label = auto_label(image_directory + filename, text_prompt)
        save_label(label, label_directory + filename + ".json")
        labeled_image_file = labeled_image_directory + filename
        display_labeled_image(Image.open(image_directory + filename), label, labeled_image_file)
