import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from COMP472.SOEN472_A1 import read_documents

vector = CountVectorizer()


def find_best_nb(x_train_tfidf, labels):
    print("------------------------------------------------------------------------------------")
    print("Looking for the best parameters for Naives Bayes...")
    decision_tree = MultinomialNB()

    param_dict = {
        "alpha": [x * 0.1 for x in range(1, 100)],
        "fit_prior": ['true', 'false']
    }
    grid = GridSearchCV(decision_tree,
                        param_grid=param_dict,
                        cv=10,
                        verbose=1,
                        n_jobs=-1)
    grid.fit(x_train_tfidf, labels)

    print("Best parameters for Naives Bayes found! ")
    print(grid.best_params_)
    print("With an accuracy of: " + str(grid.best_score_))
    print("Call with:")
    print(grid.best_estimator_)
    print("------------------------------------------------------------------------------------\n\n")


def find_best_dt_param(x_train_tfidf, labels):
    print("------------------------------------------------------------------------------------")
    print("Looking for the best parameters for Decision Tree...")

    decision_tree = DecisionTreeClassifier(random_state=42)

    param_dict = {
        "criterion": ['gini', 'entropy'],
        "splitter": ['best', 'random'],
        "max_depth": range(1, 10),
        "min_samples_split": range(2, 10),
        "min_samples_leaf": range(1, 10)
    }
    grid = GridSearchCV(decision_tree,
                        param_grid=param_dict,
                        cv=2,
                        verbose=1,
                        n_jobs=-1)
    grid.fit(x_train_tfidf, labels)

    print("Best parameters for Decision Tree found! ")
    print(grid.best_params_)
    print("With an accuracy of: " + str(grid.best_score_))
    print("Call with:")
    print(grid.best_estimator_)
    print("------------------------------------------------------------------------------------\n\n")


def main(filename):
    # (Task 0)
    all_docs, all_labels = read_documents(filename + '.txt')

    # Transforming a list of list into a list of Strings
    all_docs = [' '.join(ele) for ele in all_docs]

    split_point = int(0.80 * len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]

    x_train_counts = vector.fit_transform(train_docs)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    find_best_nb(x_train_tfidf, train_labels)
    find_best_dt_param(x_train_tfidf, train_labels)


if __name__ == '__main__':
    main("all_sentiment_shuffled")
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Enter a single file name!")
