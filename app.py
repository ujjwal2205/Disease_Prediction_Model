from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import os

app = Flask(__name__)

model = None
mlb = None
df = None

def load_resources():
    global model, mlb, df
    if model is None:
        model = joblib.load("model/rf_model.pkl")
        mlb = joblib.load("model/mlb.pkl")
        df = pd.read_excel("cleaned_combined_disease_data (1) (1).xlsx")
        df['precautions'] = df['precautions'].apply(
            lambda x: x.split(';') if isinstance(x, str) else x
        )

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Disease Prediction API is live!"})

@app.route('/predict', methods=['POST'])
def predict_disease():
    load_resources()  # 👈 lazy load here

    data = request.get_json(force=True)
    symptoms = data.get("symptoms", [])

    if not isinstance(symptoms, list):
        return jsonify({"error": "Symptoms must be a list"}), 400

    cleaned = [s.lower().replace(" ", "") for s in symptoms]
    valid_symptoms = set(mlb.classes_)
    filtered = [s for s in cleaned if s in valid_symptoms]

    if len(filtered) < 3:
        return jsonify({"error": "Please provide at least 3 valid symptoms"}), 400

    input_encoded = mlb.transform([filtered])
    prediction = model.predict(input_encoded)[0]
    info = df[df['disease'] == prediction].iloc[0]

    return jsonify({
        "Predicted Disease": prediction,
        "Precautions": info['precautions'],
        "Doctor Type": info['specialist']
    })
