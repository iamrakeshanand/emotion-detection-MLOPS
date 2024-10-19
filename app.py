from flask import Flask, request, jsonify, render_template
import joblib
app = Flask(__name__)
model = joblib.load('emotion_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')
# API endpoint to predict emotion based on text input
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data in JSON format
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided for emotion detection'}), 400
    text_input = [data['text']]  
    prediction = model.predict(text_input)[0] 
    # Respond with the prediction as JSON
    return jsonify({'emotion': prediction})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

