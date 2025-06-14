import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model safely using absolute path
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = [float(data[feature]) for feature in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]
        input_array = np.array(input_data).reshape(1, -1)
        result = model.predict(input_array)[0]
        return jsonify({"prediction": "You have diabetes" if result == 1 else "You don't have diabetes"})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
