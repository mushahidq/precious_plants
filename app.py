import numpy as np
import pickle
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from flask import Flask, flash, request, redirect, url_for, send_from_directory, send_file, render_template
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = keras.models.load_model('./saved-model/')

# process image
def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        return None

def get_result(file):
    img_arr = process_image(file)
    return model.predict(img_arr)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if(os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename))):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result', filename=filename))
    else:
        if(os.path.isfile('static/images/image.png')):
            os.remove('static/images/pie.png')
    return render_template('form.html')

@app.route('/result/<filename>')
def result(filename):
    file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = get_result(file)
    return render_template('result.html', result=result)

if(__name__ == "__main__"):
    app.run()
