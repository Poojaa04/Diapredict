import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load the trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create main window
root = tk.Tk()
root.title("Diapredict")

# Window size
root.geometry("400x500")

# Labels and entries dictionary
entries = {}

# List of features
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Create labels and entries for each feature
for i, feature in enumerate(features):
    label = tk.Label(root, text=feature)
    label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=10)
    entries[feature] = entry

# Prediction function
def predict():
    try:
        # Read values from entries and convert to float
        input_data = [float(entries[feature].get()) for feature in features]
        
        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Get prediction (0 or 1)
        prediction = model.predict(input_array)[0]
        
        if prediction == 1:
            messagebox.showinfo("Result", "Prediction: You have diabetes.")
        else:
            messagebox.showinfo("Result", "Prediction: You do NOT have diabetes.")
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numeric values for all fields.")

# Predict button
predict_btn = tk.Button(root, text="Predict", command=predict)
predict_btn.grid(row=len(features), column=0, columnspan=2, pady=20)

# Start the GUI loop
root.mainloop()
