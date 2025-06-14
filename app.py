# app.py (Flask version)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = [float(data[feature]) for feature in [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]]
    input_array = np.array(input_data).reshape(1, -1)
    result = model.predict(input_array)[0]
    return jsonify({"prediction": "You have diabetes" if result == 1 else "You don't have diabetes"})

if __name__ == "__main__":
    app.run(debug=True)
