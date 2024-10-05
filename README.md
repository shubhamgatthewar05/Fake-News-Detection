

# 📰 Fake News Detection App


## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Demo](#demo)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Model Training](#model-training)
9. [Streamlit Application](#streamlit-application)
10. [Directory Structure](#directory-structure)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)
14. [Contact](#contact)

---

## Project Overview

Fake news has become a significant concern in today's digital age, influencing public opinion and decision-making. This project aims to develop a machine learning-based application that can effectively classify news articles as **Fake** or **True**. By leveraging various classifiers and natural language processing techniques, the application provides accurate predictions, helping users discern the authenticity of news content.

---

## Features

- **Multiple Classifiers**: Utilize a range of machine learning models including Logistic Regression, Decision Tree, Gradient Boosting, Random Forest, XGBoost, and Support Vector Machine (SVM) for classification.
- **Interactive Interface**: A user-friendly Streamlit interface allows users to input news text and select different machine learning models for prediction.
- **Real-time Predictions**: Get instant classification results indicating whether the news is fake or true.
- **Data Visualization**: Visualize the distribution of fake and true news and explore word clouds highlighting common terms in each category.
- **Model Evaluation**: Assess and compare the performance of different classifiers based on accuracy and detailed classification reports.

---

## Demo

![App Demo](https://i.imgur.com/abc5678.gif) *(Replace with an actual demo GIF or screenshot)*

*Here you can include a GIF or screenshot demonstrating the application's functionality.*

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`, `wordcloud`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Natural Language Processing: `nltk`
  - Web App: `Streamlit`
  - Others: `joblib`, `tqdm`
- **Environment**: Jupyter Notebook (for development), Streamlit (for deployment)

---

## Dataset

The project utilizes two primary datasets:

1. **Fake News Dataset (`Fake.csv`)**
2. **True News Dataset (`True.csv`)**

*Ensure you have these datasets in your project directory. If not, you can obtain them from [Kaggle's Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).*

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a Virtual Environment (Optional but Recommended)

It's good practice to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'env'
python -m venv env

# Activate the virtual environment
# Windows:
env\Scripts\activate

# macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies

Ensure you have `pip` installed. Then, install the required Python libraries:

```bash
pip install -r requirements.txt
```

*If you don't have a `requirements.txt` file, you can create one with the following content:*

```plaintext
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
nltk
joblib
wordcloud
streamlit
tqdm
```

Alternatively, install dependencies individually:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost nltk joblib wordcloud streamlit tqdm
```

### 4. Download NLTK Data

Run the following commands in a Python interpreter or add them to your scripts to download necessary NLTK datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Usage

### 1. Train and Save the Models

Before running the Streamlit application, you need to train the machine learning models and save them for later use.

**a. Ensure Data Availability**

Place the `Fake.csv` and `True.csv` datasets in your project directory.

**b. Run the Training Script**

Execute the `train_models.py` script to preprocess data, train models, evaluate their performance, and save the trained models along with the vectorizer.

```bash
python train_models.py
```

**Expected Output:**

The script will display classification reports for each model and save the models and vectorizer as `.pkl` files.

```plaintext
=== Logistic Regression ===
Accuracy: 0.XXXX
<Classification Report>

=== Decision Tree ===
Accuracy: 0.XXXX
<Classification Report>

...

All models and the vectorizer have been saved successfully!
```

**Generated Files:**

- `tfidf_vectorizer.pkl`
- `logistic_regression_model.pkl`
- `decision_tree_model.pkl`
- `gradient_boosting_model.pkl`
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `support_vector_machine_model.pkl`

### 2. Run the Streamlit Application

Launch the Streamlit app to interact with the Fake News Detection system.

```bash
streamlit run app.py
```

After running the above command, Streamlit will automatically open a new tab in your default web browser (usually at `http://localhost:8501`). If it doesn't open automatically, copy and paste the provided URL into your browser.

### 3. Using the Application

- **Input Text**: Enter the news article text you wish to classify in the provided text area.
- **Select Model**: Choose the machine learning model from the dropdown menu you want to use for prediction.
- **Predict**: Click the "Predict" button to receive the classification result indicating whether the news is **Fake** or **True**.

---

## Model Training

The `train_models.py` script handles data preprocessing, model training, evaluation, and saving the trained models.

### Steps Performed:

1. **Data Loading**: Reads `Fake.csv` and `True.csv` datasets.
2. **Labeling**: Assigns `0` to fake news and `1` to true news.
3. **Data Combining**: Merges the two datasets.
4. **Data Cleaning**: Removes irrelevant columns and handles missing values.
5. **Text Preprocessing**: Cleans the text data by removing unwanted characters, tokenizing, removing stopwords, and lemmatizing.
6. **Vectorization**: Converts text data into numerical features using `TfidfVectorizer`.
7. **Model Training**: Trains multiple classifiers.
8. **Model Evaluation**: Evaluates each model's performance using accuracy and classification reports.
9. **Model Saving**: Saves the trained models and vectorizer using `joblib` for later use.

---

## Streamlit Application

The `app.py` script creates an interactive web application using Streamlit, allowing users to input news text, select a model, and get predictions.

### Key Components:

- **Title & Description**: Provides an overview of the application.
- **User Input**: Text area for users to input news content.
- **Model Selection**: Dropdown menu to choose the desired classifier.
- **Prediction**: Button to execute the prediction and display results.

---

## Directory Structure

Organize your project files as follows for clarity and maintainability:

```
fake-news-detection/
├── app.py
├── train_models.py
├── requirements.txt
├── Fake.csv
├── True.csv
├── tfidf_vectorizer.pkl
├── logistic_regression_model.pkl
├── decision_tree_model.pkl
├── gradient_boosting_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
├── support_vector_machine_model.pkl
├── README.md
└── images/
    └── app_screenshot.png
```

- **`app.py`**: Streamlit application script.
- **`train_models.py`**: Script for training and saving machine learning models.
- **`requirements.txt`**: List of project dependencies.
- **`Fake.csv` & `True.csv`**: Datasets for fake and true news.
- **`.pkl` files**: Saved models and vectorizer.
- **`README.md`**: Project documentation.
- **`images/`**: Directory for storing screenshots or demo images.

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### Steps to Contribute:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Changes**

4. **Commit Your Changes**

   ```bash
   git commit -m "Add YourFeatureName"
   ```

5. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**

---


## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the datasets.
- [Streamlit](https://streamlit.io/) for enabling easy web app development.
- [NLTK](https://www.nltk.org/) for natural language processing tools.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
- [XGBoost](https://xgboost.readthedocs.io/) for the gradient boosting framework.

---

## Contact

For any questions or suggestions, feel free to reach out:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/yourprofile)
- **GitHub**: [yourusername](https://github.com/yourusername)

---

*Happy Coding! 🚀*

---

## Additional Tips

1. **Screenshots and GIFs**: Including visual aids can greatly enhance your README. Use the `images/` directory to store them and reference them using Markdown.

   ```markdown
   ![App Screenshot](images/app_screenshot.png)
   ```

2. **Badges**: Add badges to showcase build status, license, etc.

   ```markdown
   ![GitHub license](https://img.shields.io/github/license/yourusername/fake-news-detection)
   ```

3. **Live Demo**: If deployed, provide a link to the live application.

   ```markdown
   [Live Demo](https://yourapp.streamlit.app/)
   ```

4. **Installation Instructions**: Ensure clarity in installation steps to help users set up the project without issues.

5. **Consistent Formatting**: Use consistent Markdown formatting for headings, lists, and other elements to maintain readability.
