from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    image_file = request.files['image']
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=30, maxRadius=100)

    if circles is not None:
        result = "🤖 Форма на ириса – нормална"
    else:
        result = "⚠️ Неясна форма – възможен дисбаланс"

    return jsonify({'analysis': result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
