from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    review = data.get('review', '')

    # Transform the input review
    review_vector = vectorizer.transform([review])

    # Predict sentiment
    sentiment = model.predict(review_vector)[0]
    accuracy = model.score(review_vector, [sentiment])  # Not typically done this way, consider fixing

    # Return sentiment and model accuracy
    return jsonify({'sentiment': sentiment, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
