# train_models.py

import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Data Loading
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# 2. Labeling
data_fake['class'] = 0  # Fake news labeled as 0
data_true['class'] = 1  # True news labeled as 1

# 3. Combining Datasets
data = pd.concat([data_fake, data_true], axis=0).reset_index(drop=True)

# 4. Dropping Irrelevant Columns
data = data.drop(['title', 'subject', 'date'], axis=1)

# 5. Handling Missing Values
data.dropna(inplace=True)

# 6. Text Preprocessing Function
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

# 7. Applying Preprocessing
data['text'] = data['text'].apply(preprocess_text)

# 8. Feature and Label Separation
X = data['text']
y = data['class']

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 10. Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 11. Model Initialization
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
   
}

# 12. Model Training and Evaluation
model_accuracy = {}

for model_name, model in models.items():
    model.fit(X_train_vect, y_train)
    pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, pred)
    model_accuracy[model_name] = accuracy
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, pred))

# 13. Saving Models and Vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
for model_name, model in models.items():
    joblib.dump(model, f'{model_name.replace(" ", "_").lower()}_model.pkl')

print("\nAll models and the vectorizer have been saved successfully!")
