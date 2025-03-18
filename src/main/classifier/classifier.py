from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import pickle

path = 'train.csv'
# load data from CSV file
train_data = pd.read_csv(path, delimiter=';', header=None)

# extract questions and answer types
questions = train_data.iloc[:, 1]
answer_types = train_data.iloc[:, 3]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(questions, answer_types, test_size=0.2, random_state=42)

# convert questions to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# train a logistic regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_vectors, y_train)

# save the trained classifier to a file
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# save the CountVectorizer to a file
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# predict answer types for the test set
y_pred = classifier.predict(X_test_vectors)

# evaluate the classifier
print(classification_report(y_test, y_pred))
