#   /Website
#       app.py
#       /templates
#               index.html
#       /static
#           /css
#               style.css
#           /image
#               logo.png
#       /uploads
#           csv generate output: DeepSP_descriptors.csv
from flask import Flask, request, render_template, redirect, url_for,jsonify
import os
from urllib.parse import quote as url_quote
## Machine Learning
from rdkit.Chem import Draw
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

NB_etr_model = joblib.load('NB_etr_model.joblib')
G_etr_model = joblib.load('G_etr_model.joblib')
AF_etr_model = joblib.load('AF_etr_model.joblib')
CC_rf_model = joblib.load('CC_rf_model.joblib')

app = Flask(__name__)

app.secret_key = 'pkl'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt','csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    descriptor_values = {}
    for descriptor, descriptor_func in Descriptors.descList:
        try:
            value = descriptor_func(mol)
            descriptor_values[descriptor] = value
        except Exception as e:
            print(f"Error computing descriptor {descriptor}: {e}")
    return descriptor_values

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])

def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])

def upload_file():
    result = {}
    if request.method == 'POST':
        smile = request.form['smile_name']
        mole = Chem.MolFromSmiles(smile)
        if mole is not None:
            img = Draw.MolToImage(mole)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        try:
            X = compute_descriptors(smile)
            X_df = pd.DataFrame([X])
            X_values = np.array(list(X.values())).reshape(1, -1)
            #print(X)
            if np.isnan(X_values).any():
                predictions = [None, None, None, None]
            else:
                predictions = [
                    NB_etr_model.predict(X_df)[0],  # Load these models appropriately
                    G_etr_model.predict(X_df)[0],
                    AF_etr_model.predict(X_df)[0],
                    CC_rf_model.predict(X_df)[0]
                ]
            result = {
                "SMILE": smile,
                "Structure": f"data:image/png;base64,{img_base64}",
                "Normal boiling point": predictions[0],
                "Enthalpy of Formation (Gaseous state)": predictions[1],
                "Acentric factor": predictions[2],
                "Critical compressibility factor": predictions[3]
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            result = {"error": "An error occurred while processing your SMILE string."}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

