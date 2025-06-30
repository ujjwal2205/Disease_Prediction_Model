from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import os 

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model/rf_model.pkl")
mlb = joblib.load("model/mlb.pkl")
df = pd.read_excel("cleaned_combined_disease_data (1) (1).xlsx")
df['precautions'] = df['precautions'].apply(lambda x: x.split(';') if isinstance(x, str) else x)

# ✅ Uptime check route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Disease Prediction API is live!"})

@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not isinstance(symptoms, list):
        return jsonify({"error": "Symptoms must be a list"}), 400

    # ✅ Clean input
    cleaned = [s.lower().replace(" ", "") for s in symptoms]

    # ✅ Filter invalid symptoms
    valid_symptoms = set(mlb.classes_)
    filtered = [s for s in cleaned if s in valid_symptoms]

    if len(filtered) < 3:
        return jsonify({"error": "Please provide at least 3 valid symptoms"}), 400

    # ✅ Encode and predict
    input_encoded = mlb.transform([filtered])
    prediction = model.predict(input_encoded)[0]
    info = df[df['disease'] == prediction].iloc[0]

    return jsonify({
        "Predicted Disease": prediction,
        "Precautions": info['precautions'],
        "Doctor Type": info['specialist']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
