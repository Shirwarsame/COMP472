import numpy as np
import matplotlib.pyplot as plt
import sys
from codecs import open
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

vectorizer = CountVectorizer()

# Read file
def read_documents(doc_file):
    docs = []
    labels = []
    try:
        with open(doc_file, encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                docs.append(words[3:])
                labels.append(words[1])
    except FileNotFoundError:
        print("File does not exist!")
        sys.exit()
    return docs, labels

# Plotting for Bar Graph
def plot_bar(labels):
    unique_labels = sorted(list(set(labels)))
    counts = [labels.count(label) for label in unique_labels]
    plt.bar(unique_labels, counts, color=['red', 'blue', 'purple', 'green'])

    for i, v in enumerate(counts):
        plt.text(plt.xticks()[0][i] - 0.10, v + 50, str(v))

    plt.title("Frequency of Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()

# Train Naive Bayes
def train_nb(documents, labels):
    # This function trains the Naive-Bayes model with training data (documents and lables)
    # and returns a classifier. The instance of the NB used is the multinominal.
    X_train_counts = vectorizer.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf_nb = MultinomialNB().fit(X_train_tfidf, labels)
    return(clf_nb)

# Train Base Decision Tree
def train_base_DT(documents, labels):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = DecisionTreeClassifier(criterion='entropy').fit(X_train_tfidf, labels)
    return model

# Train Best Decision Tree
def train_best_DT(documents, labels):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(documents)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = DecisionTreeClassifier().fit(X_train_tfidf, labels)
    return model

def classify(document, classifier):
    # This function predicts the label of a document based on the classifier passed
    X_eval_counts = vectorizer.transform(document)
    predicted = classifier.predict(X_eval_counts)
    return predicted

def accuracy(true_labels, guessed_labels, eval_labels):
    # This function evaluates the accuracy of a classifier by compated the true labels to the guessed labels
    evaluation = np.mean(true_labels == guessed_labels)
    metric = metrics.classification_report(eval_labels, guessed_labels,list(set(eval_labels)))
    return evaluation, metric

def report(filename, eval_labels, prediction):
    f = open(filename + ".txt", "w")
    eval1, metric1 = accuracy(eval_labels, prediction, eval_labels)
    f.write('---------------\n' + filename +'\n' + '---------------\n\n')
    f.write('---------------\n' + 'Accuracy\n' + '---------------\n')
    f.write(str(eval1) + '\n\n')
    f.write('---------------\n' + 'Performance Analysis\n' + '---------------\n')
    f.write(str(metric1) + '\n\n')
    f.write('---------------\n' + 'Confusion Matrix\n' + '---------------\n')
    f.write(str(metrics.confusion_matrix(eval_labels, prediction)) + "\n\n")
    f.write('---------------\n' + 'Evaluation\n' + '---------------\n')
    for i in range(len(prediction)):
        f.write(str(i) + ', ')
        if prediction[i] == 'pos':
            f.write('1')
        elif prediction[i] == 'neg':
            f.write('0')
        else:
            f.write('E')
        f.write('\n')
    f.close()

def main(filename):
    # (Task 0)
    all_docs, all_labels = read_documents(filename + '.txt')

    # Transforming a list of list into a list of Strings
    all_docs = [' '.join(ele) for ele in all_docs]

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    # Plot on Bar Graph (Task 1)
    plot_bar(all_labels)

    # (Task 2 & 3)
    # Naive Bayes
    clf_nb1 = train_nb(train_docs, train_labels)
    predicted_nb1 = classify(eval_docs, clf_nb1)
    report("NaiveBayes_" + filename, eval_labels, predicted_nb1)

    # Base Decision Tree
    clf_base_dt = train_base_DT(train_docs, train_labels)
    predicted_base_dt = classify(eval_docs, clf_base_dt)
    report("Base_DT_" + filename, eval_labels, predicted_base_dt)

    # Best Decision Tree
    clf_best_dt = train_best_DT(train_docs, train_labels)
    predicted_best_dt = classify(eval_docs, clf_best_dt)
    report("Best_DT_" + filename, eval_labels, predicted_best_dt)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Enter a single file name!")