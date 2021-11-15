import torch
import torch.nn as nn
from Autoencoders import AE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from Autoencoders import DatasetClass
import pandas as pd
import pickle
import scipy.misc as smp
from PIL import Image as im
import sys
import matplotlib.dates as mdates
import math
from sklearn.preprocessing import MinMaxScaler

TEST_SET = "nz/nz_features_closeness_entire_network_test.csv"
NUM_FEATURES = 2384

GAUSSIAN_MODELS_3 = "nz/gaussian_models_closeness.pkl"
GAUSSIAN_MODELS_2 = "nz/gaussian_models_closeness_2.pkl"
GAUSSIAN_MODELS_1 = "nz/gaussian_models_closeness_1.pkl"
TEST_DATA = TEST_SET

def load_gaussian_models(fileName):
    """ Loads the gaussian models """
    with open(fileName, "rb") as fp:
        gaussian_models = pickle.load(fp)
        return gaussian_models

def load_test_data(fileName):
    """ Loads the test data """
    test_data = np.loadtxt(fileName, delimiter=",", usecols=range(1, NUM_FEATURES + 1), skiprows=1, dtype=np.float32)
    labels = np.loadtxt(fileName, delimiter=",", usecols=[0], skiprows=1, dtype=np.unicode_)
    print(test_data)
    nodes_list = np.loadtxt(fileName, delimiter=",", usecols=range(1, NUM_FEATURES + 1), max_rows=1, dtype=np.unicode_)
    print(nodes_list)
    test_data, labels = sort_data(test_data, labels)
    return test_data, labels, nodes_list

def sort_data(test_data, labels):
    """ Sorts the data by date """
    sorted_data = []
    for i in range(len(labels)):
        date_string = labels[i][2:19]
        data = test_data[i]
        datetime_object = datetime.strptime(date_string, '%m-%d-%y %H:%M:%S')
        sorted_data.append({'date': datetime_object, 'data': data})
    sorted_data.sort(key = lambda x:x['date'])
    new_labels = []
    new_data = []
    for i in range(len(sorted_data)):
        new_labels.append(sorted_data[i]['date'])
        new_data.append(sorted_data[i]['data'])
    return new_data, new_labels

def predict_gaussian(gaussian_models, test_data, labels):
    """ Predicst for test instances anomaly scores """
    length_test_data = len(test_data)
    all_scores = [] # List of all scores
    image = []
    for row in range(length_test_data):
        total_score = 0
        image_features = []
        for col in range(NUM_FEATURES):
            # Getting the gaussian model from the corresponding column and test item
            gmm = gaussian_models[col]
            test_item = test_data[row][col]
            score = predict_item(gmm, test_item)
            total_score += score
            # Add the score to the image of this feature
            image_features.append(score)
        all_scores.append(total_score)
        image.append(image_features)
    all_scores = inverse_scores(all_scores)
    return all_scores, labels

def predict_item_all(gmm, test_items):
    """ Predicts the item for a gmm """
    reshaped_test_items = []
    for test_item in test_items:
        reshaped_test_items.append(test_item.reshape(1, -1))
    total_score = 0
    logged_scores = gmm.score_samples(reshaped_test_items)
    for i in len(logged_scores):
        total_score += logged_scores[i]
    return total_score

def predict_item(gmm, test_item):
    """ Predicts the anomaly score for an item """
    test_item = test_item.reshape(1, -1)
    return gmm.score_samples(test_item)


def inverse_scores(scores):
    """ Inverses the scores for the plotting"""
    scaler = MinMaxScaler().fit(scores)
    total = scaler.data_max_ + scaler.data_min_
    for score in scores:
        score[0] = total - score[0]
    return scores

def scale_gmm_scores(ae_scores, gmm_scores):
    """ Scales the GMM scores according to the ae scores """
    np_array_ae = np.array(ae_scores)
    # Find min and max values
    max_ae = np_array_ae.max()

    np_array_gmm = np.array(gmm_scores)
    # Find min and max values
    max_gmm = np_array_gmm.max()
    min_gmm = np_array_gmm.min()

    new_gmm_scores = []
    for score in gmm_scores:
        new_score = max_ae * (score - min_gmm) / (max_gmm - min_gmm)
        new_gmm_scores.append(new_score)
    return new_gmm_scores

def generate_scatter_plot(gmm_scores_3, gmm_scores_2, gmm_scores_1, labels):
    """
    Generates a scatter plot
    """
    # g1 = (labels, ae_scores)
    # gmm_scores = scale_gmm_scores(ae_scores, gmm_scores)
    g1 = (labels, gmm_scores_1)
    g2 = (labels, gmm_scores_2)
    g3 = (labels, gmm_scores_3)

    data = (g1, g2, g3)
    colors = ("red", "green", "black")
    groups = ("1 Components", "2 Components", "3 Components")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('GMM Components')
    plt.xlabel("Time (UTC) on day 30-08-2020")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    """ Loads the model and predicts the dataset """
    # GMM
    gaussian_models_3 = load_gaussian_models(GAUSSIAN_MODELS_3)
    test_data, labels, nodes_list = load_test_data(TEST_DATA)
    gmm_scores_3, labels = predict_gaussian(gaussian_models_3, test_data, labels)

    gaussian_models_2 = load_gaussian_models(GAUSSIAN_MODELS_2)
    test_data, labels, nodes_list = load_test_data(TEST_DATA)
    gmm_scores_2, labels = predict_gaussian(gaussian_models_2, test_data, labels)

    gaussian_models_1 = load_gaussian_models(GAUSSIAN_MODELS_1)
    test_data, labels, nodes_list = load_test_data(TEST_DATA)
    gmm_scores_1, labels = predict_gaussian(gaussian_models_1, test_data, labels)

    generate_scatter_plot(gmm_scores_3, gmm_scores_2, gmm_scores_1, labels)
if __name__ == "__main__":
    main()
