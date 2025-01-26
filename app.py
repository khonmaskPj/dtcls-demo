from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'images'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Models
det_model = YOLO("bestestdt.pt")  # Detection model
cls_model = YOLO("bestestclsV2.pt")  # Classification model

CLASS_NAMES = {
    0: 'Maiyarab',
    1: 'Maric',
    2: 'Pipek',
    3: 'Samanakkha',
    4: 'Tosakanth',
    5: 'Kabangna',
    6: 'Mongkut Queen',
    7: 'Yodchai Crown',
    8: 'Radklao Plew',
    9: 'Radklao Yod',
    10: 'Hanuman',
    11: 'Macchanu',
    12: 'Nilanon',
    13: 'Nilapat',
    14: 'Ongot',
    15: 'Pali',
    16: 'Sukrip',
    17: 'Phra Ganesha',
    18: 'Phra Isuan',
    19: 'Phra Narai',
    20: 'Phra Panchasikhora',
    21: 'Phra Phrom',
    22: 'Phra Pirap',
    23: 'Phra Prakhonthap',
    24: 'Phra Witsanukam',
    25: 'Phra Rishi'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/static/results/<filename>')
def serve_result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route("/detect_and_classify", methods=["POST"])
def detect_and_classify():
    if 'images' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    img_file = request.files['images']
    
    if img_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if img_file and allowed_file(img_file.filename):
        # Secure filename and save
        filename = secure_filename(img_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(img_path)
        
        # Detection
        results = det_model(img_path)
        
        # Read image with OpenCV
        img = cv2.imread(img_path)
        
        # Store detection results
        detection_results = []
        
        # Process only the first detected object
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]  # Select only the first box
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop image
            cropped_img = img[y1:y2, x1:x2]
            
            # Save cropped image
            crop_filename = f'crop_{filename.split(".")[0]}_0.jpg'
            crop_path = os.path.join(app.config['RESULT_FOLDER'], crop_filename)
            cv2.imwrite(crop_path, cropped_img)
            
            # Classify cropped image
            cls_results = cls_model(cropped_img)
            top_result = cls_results[0].probs.top1
            top_conf = cls_results[0].probs.top1conf.item()
            
            # Prepare result
            class_name = CLASS_NAMES.get(top_result, f"Unknown Class {top_result}")
            detection_results.append({
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class_name": class_name,
                "confidence": float(top_conf),
                "cropped_image": f'/static/results/{crop_filename}'
            })
        
        return jsonify(detection_results)
    
    return jsonify({"error": "File not allowed"}), 400
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)