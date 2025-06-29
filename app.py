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

@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if len(symptoms) < 3:
        return jsonify({"error": " Please provide at least 3 symptoms"}), 400

    # Clean & encode
    cleaned = [s.lower().replace(" ", "") for s in symptoms]
    input_encoded = mlb.transform([cleaned])

    # Predict
    prediction = model.predict(input_encoded)[0]
    info = df[df['disease'] == prediction].iloc[0]

    return jsonify({
        " Predicted Disease": prediction,
        " Precautions": info['precautions'],
        " Doctor Type": info['specialist']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
