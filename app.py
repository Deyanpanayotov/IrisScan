from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 IrisScan API is working!"

@app.route('/analyze', methods=['POST'])
def analyze():
    # Extract image URL from the JSON payload
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    # Fetch image from the URL
    try:
        resp = requests.get(image_url)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Failed to load image: {str(e)}'}), 500

    # Convert to grayscale and detect circles (iris-like structures)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
