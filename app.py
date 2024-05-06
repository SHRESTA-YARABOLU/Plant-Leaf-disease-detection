from flask import Flask, render_template, request
from keras.models import load_model
from model import predict_disease
import os

app = Flask(__name__)

model = None
disease_classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                   'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Target_Spot', 'Tomato_mosaic_virus',
                   'Tomato_healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if request.method == 'POST':
        file = request.files['file']
        file.save('temp_image.jpg')
        if model is None:
            model = load_model('model.keras')  
        prediction = predict_disease(model, 'temp_image.jpg', disease_classes)
        os.remove('temp_image.jpg')
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
