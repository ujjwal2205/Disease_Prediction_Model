from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
<<<<<<< HEAD
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

=======
import os 

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model/rf_model.pkl")
mlb = joblib.load("model/mlb.pkl")
df = pd.read_excel("cleaned_combined_disease_data (1) (1).xlsx")
df['precautions'] = df['precautions'].apply(lambda x: x.split(';') if isinstance(x, str) else x)

# ✅ Uptime check route
>>>>>>> d821ebe284b81b906b737e7937b342f8313989fc
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Disease Prediction API is live!"})

@app.route('/predict', methods=['POST'])
def predict_disease():
<<<<<<< HEAD
    load_resources()  # 👈 lazy load here

    data = request.get_json(force=True)
=======
    data = request.get_json()
>>>>>>> d821ebe284b81b906b737e7937b342f8313989fc
    symptoms = data.get("symptoms", [])

    if not isinstance(symptoms, list):
        return jsonify({"error": "Symptoms must be a list"}), 400

<<<<<<< HEAD
    cleaned = [s.lower().replace(" ", "") for s in symptoms]
=======
    # ✅ Clean input
    cleaned = [s.lower().replace(" ", "") for s in symptoms]

    # ✅ Filter invalid symptoms
>>>>>>> d821ebe284b81b906b737e7937b342f8313989fc
    valid_symptoms = set(mlb.classes_)
    filtered = [s for s in cleaned if s in valid_symptoms]

    if len(filtered) < 3:
        return jsonify({"error": "Please provide at least 3 valid symptoms"}), 400

<<<<<<< HEAD
=======
    # ✅ Encode and predict
>>>>>>> d821ebe284b81b906b737e7937b342f8313989fc
    input_encoded = mlb.transform([filtered])
    prediction = model.predict(input_encoded)[0]
    info = df[df['disease'] == prediction].iloc[0]

    return jsonify({
        "Predicted Disease": prediction,
        "Precautions": info['precautions'],
        "Doctor Type": info['specialist']
    })
<<<<<<< HEAD
=======


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
>>>>>>> d821ebe284b81b906b737e7937b342f8313989fc
