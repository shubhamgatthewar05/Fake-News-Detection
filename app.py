# app.py

import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the trained models
models = {
    "Logistic Regression": joblib.load('logistic_regression_model.pkl'),
    "Decision Tree": joblib.load('decision_tree_model.pkl'),
    "Gradient Boosting": joblib.load('gradient_boosting_model.pkl'),
    "Random Forest": joblib.load('random_forest_model.pkl'),
    "XGBoost": joblib.load('xgboost_model.pkl'),
    "Support Vector Machine": joblib.load('support_vector_machine_model.pkl')
}

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove text in brackets
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)
    
    # Remove words containing digits
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Prediction function
def predict(text, model):
    processed_text = preprocess_text(text)
    vect_text = vectorizer.transform([processed_text])
    prediction = model.predict(vect_text)[0]
    return "Fake News" if prediction == 0 else "True News"

# Streamlit App
st.title("📰 Fake News Detection App")
st.write("""
    This application uses various machine learning models to detect whether a news article is **Fake** or **True**.
    """)

# User Input
user_input = st.text_area("Enter the news text you want to classify:", height=250)

# Model Selection
model_option = st.selectbox(
    "Select the Machine Learning Model:",
    ("Logistic Regression", "Decision Tree", "Gradient Boosting", "Random Forest", "XGBoost", "Support Vector Machine")
)

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        model = models[model_option]
        prediction = predict(user_input, model)
        st.success(f"The news is classified as: **{prediction}**")
