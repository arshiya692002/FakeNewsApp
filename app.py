from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf.pkl')
accuracy = "94.6%"  # Update with actual value

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    transformed = vectorizer.transform([news])
    pred = model.predict(transformed)[0]
    result = "Real" if pred == 1 else "Fake"
    return render_template('index.html', prediction=result, accuracy=accuracy)

if __name__ == "__main__":
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


