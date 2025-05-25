from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Step 1: Create a larger sample dataset with 15 different crops and their recommended fertilizers
data = {
    'Nitrogen': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'Temperature': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
    'Rainfall': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
    'Crop': ['Wheat', 'Rice', 'Maize', 'Soybean', 'Corn', 'Barley', 'Oats', 'Sunflower', 'Cotton', 'Potato', 'Carrot', 'Beetroot', 'Lentil', 'Pea', 'Mustard'],
    'Fertilizer': ['Urea', 'DAP', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea', 'Urea']
}

df = pd.DataFrame(data)

# Step 2: Split the data into features and target
X = df[['Nitrogen', 'Temperature', 'Rainfall']]
y = df['Crop']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Function to predict the best crop and fertilizer
def predict_crop_and_fertilizer(nitrogen, temperature, rainfall):
    input_data = np.array([[nitrogen, temperature, rainfall]])
    predicted_crop = model.predict(input_data)[0]
    fertilizer = df[df['Crop'] == predicted_crop]['Fertilizer'].values[0]
    return predicted_crop, fertilizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        nitrogen = float(request.form['nitrogen'])
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        
        if not (0 <= nitrogen <= 200) or not (0 <= temperature <= 50) or not (0 <= rainfall <= 1000):
            raise ValueError("Values are out of expected range.")
        
        best_crop, best_fertilizer = predict_crop_and_fertilizer(nitrogen, temperature, rainfall)
        return jsonify({'best_crop': best_crop, 'best_fertilizer': best_fertilizer})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)