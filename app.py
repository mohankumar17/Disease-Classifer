from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

##########################################################################################
app=Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

##########################################################################################

rice_leaf_model = load_model("models/rice-leaf-model.h5")
maize_leaf_model = load_model("models/maize-leaf-model.h5")
brain_tumor_model = load_model("models/brain-tumor-model.h5")
diabetic_retino_model = load_model("models/diabetic-retino-model.h5")
malaria_cell_model = load_model("models/malaria-cell-model.h5")
lung_cancer_model = load_model("models/lung-cancer-model.h5")
blood_cell_model = load_model("models/blood-cell-model.h5")

##########################################################################################

def rice_leaf_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="Bacterial Blight"
    elif preds==1:
        preds="Brown spot"
    elif preds==2:
        preds="Leaf smut"

    return preds

#-------------------------------------------------------------------------------------#
def maize_leaf_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="Blight"
    elif preds==1:
        preds="Common Rust"
    elif preds==2:
        preds="Gray Leaf Spot"
    elif preds==3:
        preds="Healthy"

    return preds

#-------------------------------------------------------------------------------------#
def brain_tumor_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="Glioma Tumor"
    elif preds==1:
        preds="Meningioma Tumor"
    elif preds==2:
        preds="No Tumor"
    elif preds==3:
        preds="Pituitary Tumor"

    return preds

#-------------------------------------------------------------------------------------#
def diabetic_retino_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="No (Normal Eye)"
    else:
        preds="Yes (Eye with Diabetic Retino)"

    return preds

#-------------------------------------------------------------------------------------#
def malaria_cell_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds='Infected'
    else:
        preds='Uninfected'

    return preds

#-------------------------------------------------------------------------------------#
def lung_cancer_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds='Adeno Carcinoma'
    elif preds==1:
        preds='Large Cell Carcinoma'
    elif preds==2:
        preds='Normal'
    elif preds==3:
        preds='Squamous Cell Carcinoma'

    return preds

#-------------------------------------------------------------------------------------#
def blood_cell_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds='EOSINOPHIL'
    elif preds==1:
        preds='LYMPHOCYTE'
    elif preds==2:
        preds='MONOCYTE'
    elif preds==3:
        preds='NEUTROPHIL'

    return preds

###########################################################################################

@app.route("/",methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/riceleaf")
def riceleaf():
    return render_template("riceleaf.html")

@app.route("/maizeleaf")
def maizeleaf():
    return render_template("maizeleaf.html")

@app.route("/braintumor")
def braintumor():
    return render_template("braintumor.html")

@app.route("/diabetic_retino")
def diabetic_retino():
    return render_template("diabetic_retino.html")

@app.route("/malariacell")
def malariacell():
    return render_template("malariacell.html")

@app.route("/lungcancer")
def lungcancer():
    return render_template("lungcancer.html")

@app.route("/bloodcell")
def bloodcell():
    return render_template("bloodcell.html")

###########################################################################################

#----------------------------------------------------------------------------------#
@app.route('/predict_rice', methods=['GET','POST'])

def upload_rice():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = rice_leaf_model_predict(file_path, rice_leaf_model)
        result = preds
        return result
    return None

#----------------------------------------------------------------------------------#
@app.route('/predict_maize', methods=['GET','POST'])

def upload_maize():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = maize_leaf_model_predict(file_path, maize_leaf_model)
        result = preds
        return result
    return None


#----------------------------------------------------------------------------------#
@app.route('/predict_brain', methods=['GET','POST'])

def upload_brain():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = brain_tumor_model_predict(file_path, brain_tumor_model)
        result = preds
        return result
    return None


#----------------------------------------------------------------------------------#
@app.route('/predict_diabetic_retino', methods=['GET','POST'])

def upload_diab():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = diabetic_retino_model_predict(file_path, diabetic_retino_model)
        result = preds
        return result
    return None

#----------------------------------------------------------------------------------#
@app.route('/predict_malariacell', methods=['GET','POST'])

def upload_cell():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = malaria_cell_model_predict(file_path, malaria_cell_model)
        result = preds
        return result
    return None

#----------------------------------------------------------------------------------#
@app.route('/predict_lung', methods=['GET','POST'])

def upload_lung():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = lung_cancer_model_predict(file_path, lung_cancer_model)
        result = preds
        return result
    return None

#----------------------------------------------------------------------------------#
@app.route('/predict_bloodcell', methods=['GET','POST'])

def upload_blood():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = blood_cell_model_predict(file_path, blood_cell_model)
        result = preds
        return result
    return None

###########################################################################################

if __name__ == "__main__":
    app.run(debug=True)
