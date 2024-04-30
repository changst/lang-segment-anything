import warnings
import numpy as np
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
from flask import Flask, request, jsonify
import base64

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8866)