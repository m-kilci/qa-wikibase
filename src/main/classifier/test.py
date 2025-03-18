import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

path = 'test.csv'
# load the saved classifier from the file
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# load the test questions
test_data = pd.read_csv(path, delimiter=';', header=None)
test_questions = test_data.iloc[:, 1]

# load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# transform the test questions into feature vectors using the same vectorizer used during training
X_test_vectors = vectorizer.transform(test_questions)

# predict the answer types for the test questions
y_pred = classifier.predict(X_test_vectors)

# print the predicted answer types
print("Predicted Answer Types:")
print(y_pred)

# ground truth labels
true_labels = test_data.iloc[:, 3]

# evaluate the predictions
print("\nClassification Report:")
print(classification_report(true_labels, y_pred))
