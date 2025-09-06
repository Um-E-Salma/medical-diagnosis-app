import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import pickle
import json
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Heart disease feature names (based on your dataset)
FEATURE_NAMES = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# Heart disease risk interpretation
RISK_LEVELS = {
    0: {
        "risk": "Low Risk",
        "message": "Based on the provided data, you have a low risk of heart disease. However, maintain a healthy lifestyle.",
        "color": "#28a745"  # Green
    },
    1: {
        "risk": "High Risk", 
        "message": "Based on the provided data, you may have a higher risk of heart disease. Please consult with a healthcare professional.",
        "color": "#dc3545"  # Red
    }
}

def load_model_from_registry(model_name, model_version=None):
    """Load model from MLflow Model Registry."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://ec2-16-16-192-128.eu-north-1.compute.amazonaws.com:5000/")
        
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest"
            
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from registry: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        return None

def load_local_model(model_path):
    """Load model from local pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from local file: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading local model: {e}")
        return None

def preprocess_input(data):
    """Preprocess input data for heart disease prediction."""
    try:
        # Convert input to DataFrame with proper feature names
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Multiple predictions
            df = pd.DataFrame(data)
        else:
            raise ValueError("Input data must be dict or list of dicts")
        
        # Ensure all required features are present
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                df[feature] = 0  # Default value
        
        # Reorder columns to match training data
        df = df[FEATURE_NAMES]
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df.values
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

# Initialize the model
try:
    # Try loading from MLflow registry first
    model = load_model_from_registry("heart_disease_model", "1")
    if model is None:
        # Fallback to local model
        model = load_local_model("./rf_model.pkl")
    
    if model is None:
        raise Exception("Could not load model from registry or local file")
        
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "Heart Disease Prediction API",
        "status": "active",
        "model_loaded": model is not None
    })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_status": "loaded" if model else "not_loaded"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        patient_data = data.get('patient_data')
        
        if not patient_data:
            return jsonify({"error": "No patient data provided"}), 400

        # Preprocess the input data
        processed_data = preprocess_input(patient_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Get risk information
        risk_info = RISK_LEVELS[prediction]
        
        response = {
            "prediction": int(prediction),
            "risk_level": risk_info["risk"],
            "message": risk_info["message"],
            "confidence": {
                "low_risk": round(float(prediction_proba[0]) * 100, 2),
                "high_risk": round(float(prediction_proba[1]) * 100, 2)
            },
            "input_data": patient_data
        }
        
        logger.info(f"Prediction made: {prediction}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict heart disease risk for multiple patients."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        patients_data = data.get('patients_data')
        
        if not patients_data:
            return jsonify({"error": "No patients data provided"}), 400

        # Preprocess the input data
        processed_data = preprocess_input(patients_data)
        
        # Make predictions
        predictions = model.predict(processed_data)
        predictions_proba = model.predict_proba(processed_data)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            risk_info = RISK_LEVELS[pred]
            results.append({
                "patient_id": i + 1,
                "prediction": int(pred),
                "risk_level": risk_info["risk"],
                "message": risk_info["message"],
                "confidence": {
                    "low_risk": round(float(proba[0]) * 100, 2),
                    "high_risk": round(float(proba[1]) * 100, 2)
                }
            })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/generate_risk_chart', methods=['POST'])
def generate_risk_chart():
    """Generate a risk distribution chart."""
    try:
        data = request.get_json()
        predictions = data.get('predictions')
        
        if not predictions:
            return jsonify({"error": "No predictions provided"}), 400

        # Count risk levels
        risk_counts = {"Low Risk": 0, "High Risk": 0}
        for pred in predictions:
            if pred == 0:
                risk_counts["Low Risk"] += 1
            else:
                risk_counts["High Risk"] += 1

        # Generate pie chart
        plt.figure(figsize=(8, 6))
        colors = ['#28a745', '#dc3545']  # Green for low risk, Red for high risk
        
        plt.pie(
            risk_counts.values(),
            labels=risk_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        plt.title('Heart Disease Risk Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')

        # Save chart to BytesIO
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', bbox_inches='tight', dpi=300)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_feature_importance', methods=['GET'])
def generate_feature_importance():
    """Generate feature importance chart."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get feature importance (assuming RandomForest model)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame for plotting
            feature_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Generate horizontal bar chart
            plt.figure(figsize=(10, 8))
            plt.barh(feature_df['Feature'], feature_df['Importance'], 
                    color='steelblue', alpha=0.7)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Heart Disease Prediction - Feature Importance', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save chart to BytesIO
            img_io = io.BytesIO()
            plt.savefig(img_io, format='PNG', bbox_inches='tight', dpi=300)
            img_io.seek(0)
            plt.close()
            
            return send_file(img_io, mimetype='image/png')
        else:
            return jsonify({"error": "Model does not support feature importance"}), 400
            
    except Exception as e:
        logger.error(f"Feature importance chart generation failed: {e}")
        return jsonify({"error": f"Feature importance generation failed: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        info = {
            "model_type": type(model).__name__,
            "features": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
            "prediction_classes": ["Low Risk (0)", "High Risk (1)"]
        }
        
        # Add model-specific info if available
        if hasattr(model, 'n_estimators'):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info["max_depth"] = model.max_depth
            
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": f"Error getting model info: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
