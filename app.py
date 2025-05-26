from flask import Flask, request, jsonify
import os
from PIL import Image
import io
from flask_cors import CORS
from ultralytics import YOLO
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load your YOLO liveness detection model
model = pickle.load(open("model.pkl","rb"))  # Replace with your model path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Get image from form data
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Save temporarily
    temp_path = 'temp_image.jpg'
    img.save(temp_path)

    # Run inference
    results = model(temp_path)

    # Process results
    result_data = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            result_data.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({
        'predictions': result_data,
        'message': f'Found {len(result_data)} objects'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)