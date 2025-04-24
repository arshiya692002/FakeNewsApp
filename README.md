Here’s an updated version of your README that reflects the use of **Render** for frontend hosting, **Python** for backend development, and **Logistic Regression** in your Jupyter notebook file:

---

# Fake News Detection System

A machine learning-based web application that detects whether a given news article is real or fake. The project utilizes advanced Natural Language Processing (NLP) techniques along with Logistic Regression for text classification.

## 📑 Project Overview

This repository contains a **Fake News Detection System** built using **Flask** for the backend and **HTML** for the frontend. The machine learning model is trained on a dataset of news articles and uses **Logistic Regression** to classify news articles as real or fake. The system is deployed with **Render** and is fully functional through a simple web interface.

### Key Features:
- **Logistic Regression Model**: The project uses Logistic Regression for text classification.
- **Multilingual Support**: The app can translate non-English news to English using Google Translate before prediction.
- **Web Interface**: Developed with **Flask** and **HTML**, the user can input news articles to predict their authenticity.
- **Model Confidence**: Displays the confidence level of the model’s prediction (real or fake).
- **Export Results**: Option to export the results as a PDF.
- **Real-time News Testing**: Instantaneous predictions are made when a news article is submitted.

## ⚙️ Technologies Used

- **Frontend**: HTML, Bootstrap (hosted on Render)
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Logistic Regression
- **Text Processing**: TF-IDF for feature extraction
- **Translation**: Google Translate API for multilingual support
- **Modeling**: Logistic Regression, trained on news dataset

## 🚀 Features

- **Fake News Detection**: Predict whether a given news article is real or fake.
- **Automatic Translation**: Non-English news articles are automatically translated to English before prediction.
- **Real-time Feedback**: Immediate results displayed after submitting news text.
- **Beautiful UI**: A user-friendly and responsive interface built with Bootstrap.
- **Prediction Explanation**: The app displays the original and translated text (if applicable), along with the prediction (Real or Fake).
  
## 📂 Folder Structure

```
FakeNewsApp/
│
├── app.py                  # Main Flask application file
├── static/                 # CSS, JS, images for the front-end
│   ├── style.css
│   └── script.js
├── templates/              # HTML files for the UI
│   ├── index.html
│   └── result.html
├── models/                 # Trained models
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
├── data/                   # Dataset used for training
│   └── Fake.csv
│   └── True.csv
├── requirements.txt        # Dependencies file
└── README.md               # This file
```

## 📈 How to Run

### Prerequisites

- Python 3.x
- Pip (Python's package installer)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/arshiya692002/FakeNewsApp.git
   cd FakeNewsApp
   ```

2. **Install Required Libraries:**

   It’s recommended to use a virtual environment to avoid conflicts with other Python projects.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   To start the Flask web application, use the following command:

   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/` to start using the Fake News Detection System.

### Dataset

This project uses the **Fake News Dataset** which contains labeled data for news articles. The dataset is available in the repository files as **Fake.csv** and **True.csv**.

## 🧠 Model Training

- **Data Preprocessing**: Text data is cleaned by removing stop words, punctuation, and applying stemming/lemmatization.
- **Feature Extraction**: The **TF-IDF** vectorizer transforms the text data into numerical vectors.
- **Model**: A **Logistic Regression** model is trained to classify news articles as real or fake based on the features.
- **Evaluation**: The model is evaluated based on accuracy, precision, recall, and F1-score.

## 🖥 Deployment

The application is deployed on **Render**, a platform for hosting web applications. The Flask backend serves the model, and the frontend is hosted via **HTML**.

## 📑 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
