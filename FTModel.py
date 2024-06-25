import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDOneClassSVM
from sklearn.metrics import confusion_matrix, det_curve, roc_curve, precision_recall_curve
import pickle

from math import ceil
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.sparse import csc_matrix
from typing import Literal

def set_vectorizer(v_type: Literal["CountVectorizer", "HashingVectorizer", "TfidfTransformer", "TfidfVectorizer"], v_params: dict | None = None):
    '''Returns given text vectorizer with given parameters'''
    vectorizers = {"CountVectorizer": CountVectorizer, "HashingVectorizer": HashingVectorizer,
                   "TfidfTransformer": TfidfTransformer, "TfidfVectorizer": TfidfVectorizer}
    return vectorizers[v_type]().set_params(**v_params) if v_params != None else vectorizers[v_type]()

def set_classifier(c_type: Literal["LogisticRegression", "LogisticRegressionCV", "PassiveAggressiveClassifier", "Perceptron", "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SGDOneClassSVM"], c_params: dict | None = None):
    '''Returns given classifier with given parameters'''
    classifiers = {"LogisticRegression": LogisticRegression, "LogisticRegressionCV": LogisticRegressionCV,
                   "PassiveAggressiveClassifier": PassiveAggressiveClassifier, "Perceptron": Perceptron,
                   "RidgeClassifier": RidgeClassifier, "RidgeClassifierCV": RidgeClassifierCV,
                   "SGDClassifier": SGDClassifier, "SGDOneClassSVM": SGDOneClassSVM}
    return classifiers[c_type]().set_params(**c_params) if c_params != None else classifiers[c_type]()

def build_model(df: pd.DataFrame, x_column: str | list, y_column: str,
                v_type: Literal["CountVectorizer", "HashingVectorizer", "TfidfTransformer", "TfidfVectorizer"],
                c_type: Literal["LogisticRegression", "LogisticRegressionCV", "PassiveAggressiveClassifier", "Perceptron", "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SGDOneClassSVM"],
                v_params: dict | None = None, c_params: dict | None = None, file_name: str = "model.pickle", random_state: int | None = None, verbose: bool = False):
    '''Builds a PassiveAggressiveClassifier model with a Tfidf Vectorizer and returns the training and test sets.
    Also creates a pickle file. Loading the file will return a tupple: (vectorizer, classifier, labels, feature_names)

    Parameters
    ----------
    df - pandas DataFrame containing the dataset
    
    x_column - the label (or list of labels) of the input data

    y_column - the label of the target data

    v_type - text vectorizer type to build

    c_type - classifier type to build
    
    v_params - a dictionary of vectorizer parameter
    
    c_params - a dictionary of classifier parameter
    
    file_name - the name of the pickle file to save the model to

    random_state - random state to use in train_test_split for testing
    
    verbose - enables debugging output'''

    #Sort values to get the same order as the one that will be used in the classifier
    labels = df[y_column].sort_values().unique()
    X_train, X_test, y_train, y_test = train_test_split(df[x_column], df[y_column], random_state = random_state)

    #Set and fit vectorizier, vectorize input
    vectorizer = set_vectorizer(v_type, v_params)
    time_start = time()
    X_train = vectorizer.fit_transform(X_train)
    train_time = time() - time_start
    time_start = time()
    X_test = vectorizer.transform(X_test)
    test_time = time() - time_start
    feature_names = vectorizer.get_feature_names_out()

    #Set and fit classifier
    classifier = set_classifier(c_type, c_params)
    time_start = time()
    classifier.fit(X_train, y_train)
    fit_time = time() - time_start

    #Record model
    model = (vectorizer, classifier, labels, feature_names)
    with open(file_name, "wb") as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

    #Debug output
    if verbose:
        print(f"Classifier {c_type} Model {file_name}:")
        print(f"    Training data vectorized in {train_time:.3f}s")
        print(f"        n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(f"    Testing data vectorized in {test_time:.3f}s")
        print(f"        n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")
        print(f"    Model fitted in {fit_time:.3f}s")
        print(f"    Average model accuracy: {classifier.score(X_test, y_test):.3f}")

    return X_train, X_test, y_train, y_test

def output_graphs_generic(classifier, X_train: csc_matrix, X_test: csc_matrix, y_test: pd.Series, labels: np.ndarray, feature_names: np.ndarray):
    '''Outputs plotly graphs for given model and data. Doesn't output graphs for binary classification
    
    Parameters
    ----------
    classifier - model's classifier, [1] member of the model.pickle file data
    
    X_train - training input
    
    X_test - test input
    
    y_test - test target
    
    labels - labels array [2] member of the model.pickle file data
    
    feature_names - feature names array, [3] member of the model.pickle file data'''
    pred = classifier.predict(X_test)
    data = confusion_matrix(y_test, pred, labels = labels)
    fig = go.Figure()
    fig.add_heatmap(x = labels, y = labels, z = data)
    font_dict = {"size": 32, "weight": "bold"}
    xaxis_dict = {"title": {"text": "Predicted label", "font": font_dict}, "tickfont_size": 16}
    yaxis_dict = {"title": {"text": "True label", "font": font_dict}, "tickfont_size": 16}
    title_dict = {"text": "Confusion matrix", "font": font_dict}
    fig.update({"layout": {"title": title_dict, "xaxis": xaxis_dict, "yaxis": yaxis_dict}})
    fig.show()

    average_feature_effects = classifier.coef_ * np.asarray(X_train.mean(axis = 0)).ravel()

    #Binary classification
    if len(average_feature_effects) == 1:
        top_feature_effects = np.argsort(average_feature_effects[0])[-10:][::-1] #Records top 10 feature effects
        top_feature_indices = np.sort(top_feature_effects)
        top_features = feature_names[top_feature_indices]

        bottom_feature_effects = np.argsort(average_feature_effects[0])[:10][::-1] #Records bottom 10 feature effects
        bottom_feature_indices = np.sort(bottom_feature_effects)
        bottom_features = feature_names[bottom_feature_indices]

        fig = make_subplots(rows = 2, cols = 1, subplot_titles = ["Top feature effects", "Bottom feature effects"], shared_yaxes = True)
        fig.add_bar(x = top_features, y = average_feature_effects[0, top_feature_indices], row = 1, col = 1)
        fig.add_bar(x = bottom_features, y = average_feature_effects[0, bottom_feature_indices], row = 2, col = 1)
        font_dict = {"size": 16, "weight": "bold"}
        xaxis_dict["title"]["text"] = "Feature"
        yaxis_dict["title"]["text"] = "Average effect"
        title_dict["text"] = f"Average feature effect (label - {labels[1]})"
        fig.update_layout({"title": title_dict, "showlegend": False})
        fig.update_xaxes(xaxis_dict)
        fig.update_yaxes(yaxis_dict)
        fig.show()
    #Classification into 3+ classes
    else:
        top_amount = ceil(20/len(labels)) #Make sure we extract an amount of top effects that can be reasonably displayed on the graph
        for i in range(len(labels)):
            top_feature_effects = np.argsort(average_feature_effects[i])[-top_amount:][::-1] #Records top X feature effects
            if i == 0: top_feature_indices = top_feature_effects #Initial
            else: top_feature_indices = np.concatenate((top_feature_indices, top_feature_effects), axis = None) #Additional
        top_feature_indices = np.unique(top_feature_indices)
        top_features = feature_names[top_feature_indices]

        fig = go.Figure()
        for i, label in enumerate(labels): fig.add_bar(x = top_features, y = average_feature_effects[i, top_feature_indices], name = label) #Add a set of bars for each label
        if len(labels) > 10: fig.update_layout(showlegend = False) #Remove legend if there are more labels than the amount of automatically assigned colors
        font_dict = {"size": 16, "weight": "bold"}
        xaxis_dict["title"]["text"] = "Feature"
        yaxis_dict["title"]["text"] = "Average effect"
        title_dict["text"] = f"Average feature effects"
        fig.update({"layout": {"title": title_dict, "xaxis": xaxis_dict, "yaxis": yaxis_dict}})
        fig.show()

def output_graphs_binary_selection(classifier, X_test: csc_matrix, y_test: pd.Series, labels: np.ndarray):
    '''Outputs plotly graphs for given binary classification model and data.
    
    Parameters
    ----------
    classifier - model's classifier, [1] member of the model.pickle file data
    
    X_test - test input
    
    y_test - test target
    
    labels - labels array [2] member of the model.pickle file data'''
    pos_label = labels[1] #build_model records labels for binary classifiaction in following order - negative_label, positive_label
    pred = classifier.decision_function(X_test)
    data = det_curve(y_test, pred, pos_label) #data = (false_positive_rate, false_negative_rate)
    fig = go.Figure()
    fig.add_scatter(x = data[0], y = data[1])
    font_dict = {"size": 32, "weight": "bold"}
    xaxis_dict = {"title": {"text": "False positive rate", "font": font_dict}, "tickfont_size": 16}
    yaxis_dict = {"title": {"text": "False negative rate", "font": font_dict}, "tickfont_size": 16}
    title_dict = {"text": f"DET Curve (positive label - {pos_label})", "font": font_dict}
    fig.update({"layout": {"title": title_dict, "xaxis": xaxis_dict, "yaxis": yaxis_dict}})
    fig.show()

    data = roc_curve(y_test, pred, pos_label = pos_label) #data = (false_positive_rate, false_negative_rate)
    fig = go.Figure()
    fig.add_scatter(x = data[0], y = data[1])
    xaxis_dict["title"]["text"] = "False positive rate"
    yaxis_dict["title"]["text"] = "True positive rate"
    title_dict["text"] = f"ROC Curve (positive label - {pos_label})"
    fig.update({"layout": {"title": title_dict, "xaxis": xaxis_dict, "yaxis": yaxis_dict}})
    fig.show()

    #Precision - true fraction of positive predictions
    #Recall - positive fraction of true inputs
    data = precision_recall_curve(y_test, pred, pos_label = pos_label) #data = (precision, recall)
    fig = go.Figure()
    fig.add_scatter(x = data[0], y = data[1])
    xaxis_dict["title"]["text"] = "Precision"
    yaxis_dict["title"]["text"] = "Recall"
    title_dict["text"] = f"Precision-Recall Curve (positive label - {pos_label})"
    fig.update({"layout": {"title": title_dict, "xaxis": xaxis_dict, "yaxis": yaxis_dict}})
    fig.show()

def main():
    #Loads "fake_news" file that has been run through CleanData
    df = pd.read_csv("fake_news_clean.csv")
    #Sets parameters
    v_params = {"sublinear_tf": True, "max_df": 0.5, "min_df": 5, "stop_words": "english"}
    c_params = {"C": 1.1, "early_stopping": True}
    #Retrives data and model
    X_train, X_test, y_train, y_test = build_model(df, "full", "label", "TfidfVectorizer", "PassiveAggressiveClassifier", v_params, c_params, verbose = True)
    with open("model.pickle", "rb") as file:
        vectorizer, classifier, labels, feature_names = pickle.load(file)
    #Outputs score and graphs
    print(f"Score = {classifier.score(X_test, y_test):.3f}")
    output_graphs_generic(classifier, X_train, X_test, y_test, labels, feature_names)
    output_graphs_binary_selection(classifier, X_test, y_test, "REAL")

if __name__ == "__main__": main()