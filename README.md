# Six Human Emotions Detection App

## Overview

The **Six Human Emotions Detection App** is a machine learning web application built using **Python and Streamlit**.
The application analyzes user input text and predicts the underlying emotion.

The system is capable of detecting the following six emotions:

- Joy 😊
- Fear 😨
- Anger 😡
- Love ❤️
- Sadness 😢
- Surprise 😲

The model uses **TF-IDF text vectorization** and a **Logistic Regression classifier** trained on preprocessed text data.

---

## Features

- Interactive **Streamlit web interface**
- Text preprocessing using **NLTK**
- Emotion prediction using **Machine Learning**
- Displays predicted emotion and model confidence
- Easy to run locally

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- TensorFlow
- NLTK
- NumPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/emotion-detection-app.git
cd emotion-detection-app
```

Install the required dependencies:

```bash
pip install scikit-learn==1.3.2
pip install streamlit numpy nltk
pip install tensorflow==2.15.0
```

---

## Running the Application

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

After running the command, open the provided **local URL** in your browser.

---

## Model Files

The application uses the following saved model files:

- `logistic_regresion.pkl` – Logistic Regression model
- `tfidf_vectorizer.pkl` – TF-IDF text vectorizer
- `label_encoder.pkl` – Label encoder used for emotion labels

---

## How It Works

1. The user enters a sentence in the text input field.
2. The text is preprocessed using **NLTK** (cleaning, stopword removal, stemming).
3. The processed text is transformed using **TF-IDF vectorization**.
4. The trained **Logistic Regression model** predicts the emotion.
5. The predicted emotion and probability are displayed in the interface.

---

## Developers

- **Hari Krishna Tupakula**
- **Siva Sanker Pasupuleti**
- **Karthik Varma Kattika**

---

## License

This project is licensed under the **MIT License**.
