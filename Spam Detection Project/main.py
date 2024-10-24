import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify, render_template_string
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

nltk.download('stopwords')

# Load dataset
data_file_path = #enter data file path
df = pd.read_csv(data_file_path, sep='\t', header=None, names=['label', 'message'])

# Preprocess text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['message'] = df['message'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Data loaded, preprocessed, and vectorized successfully.")

# Initialize and train models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print(f"{name} Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Evaluate and visualize the Random Forest model (as an example)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Spam', 'Predicted Ham'], yticklabels=['Actual Spam', 'Actual Ham'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(y_test, y_pred, target_names=['spam', 'ham']))

# GridSearchCV for Naive Bayes
param_grid = {'alpha': np.linspace(0.1, 1.0, 10)}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha: {best_alpha}")
print(f"Best F1 Score: {grid_search.best_score_}")

nb_model_tuned = MultinomialNB(alpha=best_alpha)
nb_model_tuned.fit(X_train, y_train)
y_pred_tuned = nb_model_tuned.predict(X_test)

print(classification_report(y_test, y_pred_tuned, target_names=['spam', 'ham']))

joblib.dump(nb_model_tuned, 'naive_bayes_spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

app = Flask(__name__)

# Load the model and vectorizer for Flask app
model = joblib.load('naive_bayes_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spam Detection</title>
</head>
<body>
    <h1>Spam Detection Model</h1>
    <form action="/" method="post">
        <label for="message">Enter your message:</label>
        <input type="text" id="message" name="message" required>
        <button type="submit">Check</button>
    </form>
    {% if prediction %}
    <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        # Preprocess and vectorize the message
        message_preprocessed = preprocess_text(message)
        message_vector = vectorizer.transform([message_preprocessed])
        # Predict spam or ham
        prediction = model.predict(message_vector)[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


