from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import io

app = Flask(__name__)

# Load the trained CNN model
model = load_model(os.path.join('models', 'BreastCancerDetection.h5'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['BreastUltrasoundImage']
    if uploaded_file:
        file_contents = uploaded_file.read()
        return file_contents  # Return the file contents

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    uploaded_file_data = upload_file()

    # Process the uploaded image
    img = image.load_img(io.BytesIO(uploaded_file_data), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Process predictions as needed
    # class_labels = np.argmax(predictions, axis=1)

    return render_template('index.html', predictions=np.round(predictions,2))
    #predictions=class_labels[0]

if __name__ == '__main__':
    app.run(debug=True)
