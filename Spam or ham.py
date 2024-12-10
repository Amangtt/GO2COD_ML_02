import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df= pd.read_csv('Downloads/spam.csv')

df['spam']= df['Category'].apply(lambda x:1 if x=='spam' else 0)
x= df.drop(['Category','spam'], axis=1)
y=df['spam']
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam, test_size=0.25)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y = model.predict(X_train_vectorized)
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print(accuracy_score(y_train,y))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')