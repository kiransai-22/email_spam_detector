
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Combine vectorizer and model
from sklearn.pipeline import make_pipeline
spam_model = make_pipeline(vectorizer, model)

# Save model
os.makedirs("model", exist_ok=True)
with open(os.path.join("model", "spam_model.pkl"), "wb") as f:
    pickle.dump(spam_model, f)
