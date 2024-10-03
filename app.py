from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html file in the templates folder

# Add a route for handling the sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle user input and return the result of the sentiment analysis
    input_text = request.form['input_text']
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    return f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
