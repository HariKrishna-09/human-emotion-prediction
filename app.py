# ======================== import packages =========================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
import pickle
import nltk

# ======================== page config =============================
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

# ======================== nltk setup ==============================
try:
    stopwords = set(nltk.corpus.stopwords.words('english'))
except:
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))

# ======================== cache models ============================
@st.cache_resource
def load_models():
    lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lb = pickle.load(open('label_encoder.pkl', 'rb'))
    return lg, tfidf_vectorizer, lb

lg, tfidf_vectorizer, lb = load_models()

# ======================== text cleaning ===========================
def clean_text(text):
    stemmer = PorterStemmer()

    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stopwords]

    return " ".join(words)

# ======================== prediction ==============================
def predict_emotion(input_text):

    cleaned_text = clean_text(input_text)

    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    probabilities = lg.predict_proba(input_vectorized)[0]

    confidence = np.max(probabilities)

    return predicted_emotion, confidence, probabilities


# ======================== Sidebar =================================
st.sidebar.title("About App")
st.sidebar.write("This app detects **6 human emotions** from text.")
st.sidebar.write("Model: Logistic Regression + TF-IDF")

st.sidebar.write("### Emotions Supported")
st.sidebar.write("😊 Joy")
st.sidebar.write("😨 Fear")
st.sidebar.write("😡 Anger")
st.sidebar.write("❤️ Love")
st.sidebar.write("😢 Sadness")
st.sidebar.write("😲 Surprise")

st.sidebar.divider()

# ======================== Developers ===============================
st.sidebar.subheader("👨‍💻 Developers")

st.sidebar.write("""
- **Hari Krishna Tupakula**
- **Siva Sankar Pasupuleti**
- **Karthik Varma Kattika**
""")

st.sidebar.divider()

st.sidebar.caption("Built using Streamlit + NLP + Machine Learning")
st.sidebar.caption("Interdisciplinary Project - 2026")
st.sidebar.caption("Vignan's Foundation for Science, Technology and Research")
st.sidebar.caption("© 2026 All rights reserved")
# ======================== Main UI =================================
st.title("🧠 Human Emotion Detection")

st.markdown(
    "Detect **six human emotions from text using Natural Language Processing**"
)

st.divider()

# ======================== Example texts ===========================
st.subheader("Try Example Sentences")

col1, col2, col3 = st.columns(3)

if col1.button("Happy Example"):
    st.session_state.text = "I am feeling wonderful today"

if col2.button("Sad Example"):
    st.session_state.text = "I feel very lonely and depressed"

if col3.button("Angry Example"):
    st.session_state.text = "I am extremely angry about what happened"

# ======================== Input ===================================
user_input = st.text_area(
    "Enter your sentence",
    value=st.session_state.get("text", ""),
    height=120
)

st.divider()

# ======================== Prediction ==============================
if st.button("Predict Emotion 🚀"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        predicted_emotion, confidence, probabilities = predict_emotion(user_input)

        st.success("Prediction Completed")

        # ================= emoji mapping ===========================
        emojis = {
            "joy": "😊",
            "fear": "😨",
            "anger": "😡",
            "love": "❤️",
            "sadness": "😢",
            "surprise": "😲"
        }

        emoji = emojis.get(predicted_emotion.lower(), "")

        st.subheader(f"Predicted Emotion: {predicted_emotion} {emoji}")

        # ================= confidence bar ==========================
        st.write("### Model Confidence")

        st.progress(float(confidence))

        st.write(f"Confidence: **{round(confidence*100,2)}%**")

        st.divider()

        # ================= probability bars ========================
        st.write("### Emotion Probabilities")

        emotion_labels = lb.classes_

        for emotion, prob in zip(emotion_labels, probabilities):

            col1, col2 = st.columns([2,4])

            with col1:
                st.write(emotion.capitalize())

            with col2:
                st.progress(float(prob))