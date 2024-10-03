from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your pre-trained model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Transform the review into the format your model expects
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    
    # Create a response dictionary
    response = {
        'review': review,
        'prediction': 'Positive' if prediction[0] == 1 else 'Negative'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
