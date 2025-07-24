from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)
CORS(app)  # Enables requests from web and mobile clients like Thunkable

@app.route('/')
def home():
    return "🧠 IrisScan API is working!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Force JSON parsing and log raw input
        print("Incoming raw data:", request.get_data())
        data = request.get_json(force=True)
        print("Parsed JSON:", data)

        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'No image_url provided'}), 400

        # Fetch image from URL
        try:
            resp = requests.get(image_url)
            resp.raise_for_status()
        except Exception as e:
            return jsonify({'error': f'Failed to fetch image: {str(e)}'}), 500

        # Decode image for OpenCV
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Convert to grayscale and detect circles
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

        # Analyze result
        if circles is not None:
            result = "🤖 Форма на ириса – нормална"
        else:
            result = "⚠️ Неясна форма – възможен дисбаланс"

        return jsonify({'analysis': result})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

