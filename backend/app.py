from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None

def preprocess_text(text):
    """Clean and preprocess the input text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({"error": "No text provided"}), 400
        
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not trained yet. Please run train_model.py first"}), 500
        
        # Preprocess and vectorize the text
        cleaned_text = preprocess_text(news_text)
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = float(max(probability)) * 100
        
        result = {
            "prediction": "FAKE" if prediction == 0 else "REAL",
            "confidence": round(confidence, 2),
            "message": f"This news is likely {'FAKE' if prediction == 0 else 'REAL'} with {round(confidence, 2)}% confidence."
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    status = "ready" if (model is not None and vectorizer is not None) else "not ready"
    return jsonify({"status": status})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
