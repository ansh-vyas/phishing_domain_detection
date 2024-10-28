from flask import Flask, request, jsonify
from src.model_prediction import load_model, predict
import numpy as np

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = predict(model, features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
