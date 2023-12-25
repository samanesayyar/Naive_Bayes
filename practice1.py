import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("imdb_labelled.csv", delimiter='\t')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english') + list(punctuation)


def tokenize(text):
    # Split text into words
    tokens = word_tokenize(text)
    # Convert words to lower case
    tokens = [w.lower() for w in tokens]
    # Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # Remove tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # Filter out stop words
    words = [w for w in words if w not in stop_words and not w.isdigit()]
    # Stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed


# build the vocabulary in one pass
vocabulary = set()
for x in X:
    words = tokenize(x[0])
    vocabulary.update(words)

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(X)

print(VOCABULARY_SIZE, DOCUMENTS_COUNT)  # 2974 747

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=tokenize, vocabulary=vocabulary)

# Fit the TfIdf model
tfidf.fit(list(zip(*X))[0])

# Transform a document into TfIdf coordinates
text_tf = tfidf.transform(list(zip(*X))[0])

X_train, X_test, y_train, y_test = train_test_split(text_tf, y, test_size=0.15, random_state=123)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, y_pred))
# MultinomialNB Accuracy: 0.7964601769911505

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from baysian.Visualize import Visualize

visualize = Visualize()
visualize.save_roc(X_train=X_train, y_train=y_train, classifier=MultinomialNB(), title='ROC Of MultinomialNB',
                   classes=[0, 1])
