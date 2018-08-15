import os

from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications import imagenet_utils
import tensorflow as tf

import numpy as np

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# Initializing our Flask application and keras model
app = Flask(__name__)
model = None


# Loading the Keras model required for this API
def load_model():
    global model
    model = ResNet50(weights="imagenet")
    # this is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()


# Preprocessing and preparing the data
def prepare_image(img, target):
    # Converting the image to RGB format
    if img.mode != "RGB":
        img.convert("RGB")

    # resizing the input image and preprocessing it
    img = img.resize(target)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = imagenet_utils.preprocess_input(img)
    return img


# Homepage for landing
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# The Predict-POST request function
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("File is available at {}".format(file_path))

        # read image in PIL format
        img = image.load_img(file_path, target_size=(224, 224))

        # preparing the image for classification
        img = prepare_image(img, target=(224, 224))

        with graph.as_default():
            # classifying the input
            preds = model.predict(img)

            # Process your result for human
            pred_class = imagenet_utils.decode_predictions(preds)

            # Convert to string
            result = str(pred_class[0][0][1])
            return result
    return None


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...please wait until server has fully started")
    load_model()
    print('Model loaded. Check http://127.0.0.1:5000/')
    app.run()
