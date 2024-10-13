from flask import Flask, render_template, request 
from flask import Flask, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import time

app = Flask(__name__)
CORS(app)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('vehicle_classification_model.h5')
helmet_model = tf.keras.models.load_model('helmet_detection_model5.h5')

class_names = ['bike', 'car']
class_names_helmet = ['Without Helmet', 'With Helmet']

# ฟังก์ชันสำหรับโหลดและประมวลผลภาพ
def prepare_image(img_path, img_width=150, img_height=150):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def classify_image(img_path):
    img = prepare_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def classify_helmet(img_path):
    img = prepare_image(img_path)
    prediction = helmet_model.predict(img)
    predicted_helmet_class = np.argmax(prediction)
    helmet_confidence = np.max(prediction)
    return predicted_helmet_class, helmet_confidence

def capture_image_from_camera():
    os.makedirs('static/uploads', exist_ok=True)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Press 's' to capture an image.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow('Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            timestamp = int(time.time())
            image_path = f"static/uploads/captured_image_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return image_path

# สร้าง route สำหรับหน้า indexleo.html
@app.route('/indexleo')
def indexleo():
    return render_template('indexleo.html')

# สร้าง route สำหรับหน้า index.html
@app.route('/index', methods=['GET', 'POST'])
def index():
    result = None
    captured_image_url = None
    
    if request.method == 'POST':
        captured_image_path = capture_image_from_camera()
        
        if captured_image_path:
            predicted_class, confidence = classify_image(captured_image_path)

            if confidence < 0.9:
                result = "This is not a car or bike."
            else:
                if predicted_class == 1:
                    result = f"Prediction: Car with confidence {confidence * 100:.2f}%"
                else:
                    result = f"Prediction: Bike with confidence {confidence * 100:.2f}%"

                    
                    helmet_image_path = capture_image_from_camera()

                    if helmet_image_path:
                        predicted_helmet_class, helmet_confidence = classify_helmet(helmet_image_path)
                        if predicted_helmet_class == 1:
                            result = f"Prediction: With Helmet with confidence {helmet_confidence * 100:.2f}%"
                        else:
                            result = f"Prediction: Without Helmet with confidence {helmet_confidence * 100:.2f}%"

                        captured_image_url = captured_image_path.replace('uploads/', '')

        return render_template('index.html', result=result, captured_image=captured_image_url)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run()