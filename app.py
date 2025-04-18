from flask import Flask, request, render_template
import joblib
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# Initialize translator
translator = Translator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    # Translate non-English to English
    try:
        translated = translator.translate(news, dest='en')
        news_english = translated.text
    except:
        news_english = news  # fallback

    # Predict using model
    transformed = vectorizer.transform([news_english])
    pred = model.predict(transformed)[0]
    result = "Real" if pred == 1 else "Fake"

    return render_template('index.html',
                           prediction=result,
                           original=news,
                           translated=news_english)

if __name__ == "__main__":
    app.run(debug=True)


