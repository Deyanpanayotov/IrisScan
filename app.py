from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "🧠 IrisScan API is working!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'No image_url provided'}), 400

        resp = requests.get(image_url)
        resp.raise_for_status()

        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=100
        )

        if circles is not None:
            result = "🤖 Iris shape appears normal"
        else:
            result = "⚠️ Unclear shape — possible imbalance"

        return jsonify({'analysis': result})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


