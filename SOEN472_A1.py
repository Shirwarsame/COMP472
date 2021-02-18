import numpy as np
from codecs import open
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

# Transforming a list of list into a list of Strings
all_docs = [' '.join(ele) for ele in all_docs] 

split_point = int(0.60*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# Vectorizing the train documents
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train_docs)

# Transforming occurences into fequencies (avoid discrepancies between short and long documents)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training the NB model into a classifier - Multinominal instance
# clf_freq is using the transformed into frequencies version of the vector
# clf is using the vector of the train documents without frequency normalization
clf_freq = MultinomialNB().fit(X_train_tfidf, train_labels)
clf = MultinomialNB().fit(X_train_counts, train_labels)

# Vectorizing the evaluation documents
X_eval_counts = vectorizer.transform(eval_docs)

# Predicting the labels for the evaluation documents
# clf is using the transformed into frequencies version of the vector
# clf2 is using the vector of the train documents without frequency normalization
predicted_freq = clf_freq.predict(X_eval_counts)
predicted = clf.predict(X_eval_counts)

# Evaluating our trained model (NB - multinominal)
evaluation_freq = np.mean(predicted_freq == eval_labels)
evaluation = np.mean(predicted == eval_labels)
print(evaluation_freq)
print(evaluation)

# Printing out the metrics for the report
print(metrics.classification_report(eval_labels, predicted,list(set(eval_labels))))

#==============================================================================
#================================TODO Section==================================
#==============================================================================
# TODO:Ploting (task 1)
# TODO:Adding functions as shown in the appendex
# - def score_doc_label(document, label, classifier):
# - All the functions needed for the 2 decision tree models

#==============================================================================
#Some methods extracted from the previous code to fit the appendex construction
#==============================================================================


def train_nb(documents, labels):
    # This function trains the Naive-Bayes model with training data (documents and lables)
    # and returns a classifier. The instance of the NB used is the multinominal.
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf_nb = MultinomialNB().fit(X_train_tfidf, labels)
    return(clf_nb)


def score_doc_label(document, label, classifier):
    #TODO
    return #Probability of getting those words for a specfic label - see appendex

def classify_nb(document, classifier):
    # This function predicts the label of a document based on the classifier passed
    X_eval_counts = vectorizer.transform(document)
    predicted = classifier.predict(X_eval_counts)
    return


def accuracy(true_labels, guessed_labels):
    # This function evaluates the accuracy of a classifier by compated the true labels to the guessed labels
    evaluation = np.mean(true_labels == guessed_labels)
    metric = metrics.classification_report(eval_labels, predicted,list(set(eval_labels)))
    return evaluation, metric

# Example of the usage of the functions defined above (driver example)
clf_nb1 = train_nb(train_docs, train_labels)

predicted_nb1 = classify_nb(eval_docs, clf_nb1)
print(predicted_nb1)

eval1, metric1 = accuracy(eval_labels, predicted)
print(eval1)
print(metric1)