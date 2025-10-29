# app.py
# Main Flask application for Tumor Classification Web App
# Author: [Your Name]
# Description: End-to-end ML app with SVM/Random Forest, interactive UI, and SQLite data

from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objs as go
import plotly
import json
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
DATABASE = os.path.join('data', 'tumors.db')
MODELS_DIR = 'models'
UPLOAD_FOLDER = 'uploads'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Data Generation & Database Setup ---
def generate_data(n_rows=2000):
    np.random.seed(42)
    data = {
        'patient_id': np.arange(1, n_rows+1),
        'age': np.random.randint(20, 80, n_rows),
        'tumor_size': np.round(np.random.uniform(0.5, 5.0, n_rows), 2),
        'texture': np.round(np.random.uniform(10, 100, n_rows), 2),
        'smoothness': np.round(np.random.uniform(0.1, 1.0, n_rows), 2),
        'compactness': np.round(np.random.uniform(0.1, 1.0, n_rows), 2),
        'diagnosis': np.random.choice(['benign', 'malignant'], n_rows, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    return df

def init_db():
    if not os.path.exists(DATABASE):
        df = generate_data()
        conn = sqlite3.connect(DATABASE)
        df.to_sql('tumors', conn, index=False)
        conn.close()

init_db()

# --- Machine Learning Functions ---
def train_models():
    """Train SVM and Random Forest models and save them"""
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM tumors", conn)
    conn.close()

    # Prepare features and target
    features = ['age', 'tumor_size', 'texture', 'smoothness', 'compactness']
    X = df[features]
    y = df['diagnosis'].map({'benign': 0, 'malignant': 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate models
    svm_pred = svm_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)

    svm_accuracy = accuracy_score(y_test, svm_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Save models and scaler
    joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.pkl'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    return {
        'svm_accuracy': svm_accuracy,
        'rf_accuracy': rf_accuracy,
        'svm_report': classification_report(y_test, svm_pred, output_dict=True),
        'rf_report': classification_report(y_test, rf_pred, output_dict=True),
        'svm_cm': confusion_matrix(y_test, svm_pred).tolist(),
        'rf_cm': confusion_matrix(y_test, rf_pred).tolist()
    }

def load_models():
    """Load trained models and scaler"""
    try:
        svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        return svm_model, rf_model, scaler
    except FileNotFoundError:
        return None, None, None

def predict_tumor(features):
    """Make prediction using trained models"""
    svm_model, rf_model, scaler = load_models()
    if not all([svm_model, rf_model, scaler]):
        return None

    # Scale features
    scaled_features = scaler.transform([features])

    # Make predictions
    svm_pred = svm_model.predict(scaled_features)[0]
    rf_pred = rf_model.predict(scaled_features)[0]

    # Get probabilities
    svm_proba = svm_model.decision_function(scaled_features)[0] if hasattr(svm_model, 'decision_function') else svm_model.predict_proba(scaled_features)[0][1]
    rf_proba = rf_model.predict_proba(scaled_features)[0][1]

    return {
        'svm_prediction': int(svm_pred),
        'rf_prediction': int(rf_pred),
        'svm_probability': float(abs(svm_proba)),
        'rf_probability': float(rf_proba),
        'svm_class': 'malignant' if svm_pred == 1 else 'benign',
        'rf_class': 'malignant' if rf_pred == 1 else 'benign'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get data statistics
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM tumors", conn)
    conn.close()

    # Calculate statistics
    total_patients = len(df)
    benign_count = len(df[df['diagnosis'] == 'benign'])
    malignant_count = len(df[df['diagnosis'] == 'malignant'])

    # Check if models are trained
    models_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in ['svm_model.pkl', 'rf_model.pkl', 'scaler.pkl'])

    return render_template('dashboard.html',
                         total_patients=total_patients,
                         benign_count=benign_count,
                         malignant_count=malignant_count,
                         models_trained=models_exist)

@app.route('/train', methods=['POST'])
def train():
    try:
        results = train_models()
        flash('Models trained successfully!', 'success')
        return redirect(url_for('models'))
    except Exception as e:
        flash(f'Error training models: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/models')
def models():
    models_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in ['svm_model.pkl', 'rf_model.pkl', 'scaler.pkl'])

    if not models_exist:
        flash('Please train the models first.', 'warning')
        return redirect(url_for('dashboard'))

    # Load and display model performance
    svm_model, rf_model, scaler = load_models()

    # Get training results (you might want to save these during training)
    # For now, retrain to get metrics
    results = train_models()

    return render_template('models.html',
                         svm_accuracy=results['svm_accuracy'],
                         rf_accuracy=results['rf_accuracy'],
                         svm_report=results['svm_report'],
                         rf_report=results['rf_report'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = float(request.form['age'])
            tumor_size = float(request.form['tumor_size'])
            texture = float(request.form['texture'])
            smoothness = float(request.form['smoothness'])
            compactness = float(request.form['compactness'])

            features = [age, tumor_size, texture, smoothness, compactness]
            prediction = predict_tumor(features)

            if prediction is None:
                flash('Models not trained yet. Please train models first.', 'error')
                return redirect(url_for('predict'))

            return render_template('predict.html', prediction=prediction, features=features)

        except ValueError:
            flash('Please enter valid numerical values.', 'error')
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')

    return render_template('predict.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)

            # Process uploaded file
            try:
                df = pd.read_csv(filename)
                conn = sqlite3.connect(DATABASE)
                df.to_sql('tumors', conn, if_exists='replace', index=False)
                conn.close()

                flash('Dataset uploaded and database updated successfully!', 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
        else:
            flash('Please upload a CSV file.', 'error')

    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        features = [
            data['age'],
            data['tumor_size'],
            data['texture'],
            data['smoothness'],
            data['compactness']
        ]

        prediction = predict_tumor(features)
        if prediction is None:
            return jsonify({'error': 'Models not trained'}), 400

        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
