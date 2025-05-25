from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
import firebase_admin
from firebase_admin import credentials, db
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend


app = Flask(__name__)

# Firebase Init
# To initialize Firebase, you need to provide the path to your serviceAccountKey.json
# and your Firebase project's databaseURL.
# Replace 'path/to/your/serviceAccountKey.json' with the actual path to your file.
# Replace 'https://your-firebase-project-id.firebaseio.com/' with your database URL.
# You can find these in your Firebase project settings.
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-firebase-project-id.firebaseio.com/'
    })
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print("Please ensure 'serviceAccountKey.json' is in the correct path and 'databaseURL' is updated.")
    # If Firebase initialization fails, the application might still run,
    # but database operations will not work.

# ---
## Simulated Dataset
# This dataset is for demonstration purposes. In a real-world scenario,
# you'd load a much larger and more diverse dataset for better model performance.
data = {
    'Nitrogen': np.random.randint(50, 200, 100),
    'Temperature': np.random.uniform(20, 40, 100),
    'Rainfall': np.random.uniform(100, 800, 100),
    'pH': np.random.uniform(4.5, 8.5, 100),
    'Humidity': np.random.uniform(30, 90, 100),
    'Crop': np.random.choice(['Wheat', 'Rice', 'Maize', 'Soybean', 'Corn'], 100)
}
df = pd.DataFrame(data)
df['Fertilizer'] = df['Crop'].map({
    'Wheat': 'Urea', 'Rice': 'DAP', 'Maize': 'MOP',
    'Soybean': 'Compost', 'Corn': 'Ammonium Nitrate'
})

# ---
## Model Training
# The Decision Tree Classifier is trained here. For production,
# consider using more robust models and extensive hyperparameter tuning.
features = ['Nitrogen', 'Temperature', 'Rainfall', 'pH', 'Humidity']
X = df[features]
y = df['Crop']
model = DecisionTreeClassifier(random_state=42) # Added random_state for reproducibility
model.fit(X, y)

# ---
## Prediction Function
# This function predicts the best crop and fertilizer based on input parameters
# and generates a graph of the inputs.
def predict_crop_and_fertilizer(n, t, r, ph, h):
    input_df = pd.DataFrame([[n, t, r, ph, h]], columns=features)
    predicted_crop = model.predict(input_df)[0]
    
    # Ensure the predicted crop exists in the DataFrame to map to a fertilizer
    if predicted_crop not in df['Crop'].unique():
        # Fallback if a new crop is predicted that isn't in the original mapping
        fertilizer = "General Purpose Fertilizer" 
    else:
        fertilizer = df[df['Crop'] == predicted_crop]['Fertilizer'].values[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5)) # Set a larger figure size for better readability
    input_values = [n, t, r, ph, h]
    input_labels = ['Nitrogen', 'Temperature', 'Rainfall', 'pH', 'Humidity']
    ax.bar(input_labels, input_values, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722']) # Added more colors
    ax.set_title(f"Input Parameters for {predicted_crop} Prediction", fontsize=14)
    ax.set_ylabel("Value", fontsize=12)
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better fit
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid for better visualization
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig) # Close the plot to free up memory

    return predicted_crop, fertilizer, graph_url

# ---
## Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and convert input to float
        n = float(request.form['nitrogen'])
        t = float(request.form['temperature'])
        r = float(request.form['rainfall'])
        ph = float(request.form['ph'])
        h = float(request.form['humidity'])

        crop, fert, graph_url = predict_crop_and_fertilizer(n, t, r, ph, h)

        # Save prediction to Firebase. This will only work if Firebase was initialized successfully.
        try:
            db.reference('predictions').push({
                'timestamp': pd.Timestamp.now().isoformat(), # Add timestamp for tracking
                'nitrogen': n, 'temperature': t, 'rainfall': r,
                'ph': ph, 'humidity': h, 'crop': crop, 'fertilizer': fert
            })
        except Exception as e:
            print(f"Error saving to Firebase: {e}")
            # Continue without saving to Firebase if there's an issue

        return jsonify({
            'best_crop': crop,
            'best_fertilizer': fert,
            'graph': graph_url
        })
    except ValueError:
        return jsonify({'error': 'Invalid input. Please ensure all values are numbers.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload_csv():
    file = request.files.get('csvfile')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a valid CSV file.'}), 400

    try:
        df_upload = pd.read_csv(file)
        
        # Check if all required columns exist in the uploaded CSV
        required_columns = ['Nitrogen', 'Temperature', 'Rainfall', 'pH', 'Humidity']
        if not all(col in df_upload.columns for col in required_columns):
            return jsonify({'error': f'CSV file must contain the following columns: {", ".join(required_columns)}'}), 400

        results = []
        for index, row in df_upload.iterrows():
            # Handle potential non-numeric values in the CSV
            try:
                crop, fert, _ = predict_crop_and_fertilizer(
                    float(row['Nitrogen']), float(row['Temperature']),
                    float(row['Rainfall']), float(row['pH']), float(row['Humidity'])
                )
                results.append({'row_number': index + 1, 'crop': crop, 'fertilizer': fert})
            except ValueError:
                results.append({'row_number': index + 1, 'error': 'Invalid numeric value in row.'})
                
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
