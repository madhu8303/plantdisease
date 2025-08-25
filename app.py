from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from PIL import Image
import base64
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def process_image(image_data):
    try:
        # Load the model architecture from JSON file
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # Load model weights
        loaded_model.load_weights("model.weights.h5")

        # Define the class labels
        label = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
                "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]

        # Process the image
        img = Image.open(io.BytesIO(image_data))
        test_image = img.resize((128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        result = loaded_model.predict(test_image)
        predicted_label = label[result.argmax()]
        
        return predicted_label
    except Exception as e:
        return str(e)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'vvfgc' and password == 'vvfgc':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image_data = file.read()
    prediction = process_image(image_data)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)