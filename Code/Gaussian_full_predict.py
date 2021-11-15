import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import scipy.misc as smp
from PIL import Image as im
import sys
import matplotlib.dates as mdates
import math
from sklearn.preprocessing import MinMaxScaler
#wide
# GAUSSIAN_MODELS = "wide/gaussian_models_closeness.pkl"
# TEST_DATA = "wide/wide_features_closeness_abnormal_entire_network_day_30.csv"
# GAUSSIAN_MODELS = "nz/gaussian_models_degree.pkl"
# TEST_DATA = "nz/nz_features_degree_entire_network_test.csv"
GAUSSIAN_MODELS = "nz/gaussian_full_models_degree_1.pkl"
TEST_DATA = "nz/nz_features_degree_entire_network_test.csv"

# serbia
# TEST_DATA = "soxrs/soxrs_features_degree_abnormal_entire_network_day_30.csv"
# GAUSSIAN_MODELS = "soxrs/gaussian_models_degree.pkl"
NUM_FEATURES = 2384 # wide: 2544, nz: 2384

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

def find_index_node(nodes_list, asn):
    index = 0
    for node in nodes_list:
        node_stripped = node.replace('\n', '')
        print(node_stripped)
        if asn == node_stripped:
            return index
        index += 1
    print("ASN not found!")
    return -1

def predict(gmm, test_data, labels):
    """ Predicst for test instances anomaly scores """
    length_test_data = len(test_data)
    all_scores = [] # List of all scores
    image = []
    for row in range(length_test_data):
        image_features = []
        test_item = test_data[row]
        score = predict_item(gmm, test_item)

        image_features.append(score)
        all_scores.append(score)
        image.append(image_features)
    # construct_image(image, NUM_FEATURES)
    plot_scores(all_scores, labels)

def sort_image(image, cols):
    length_image = len(image)
    score_to_index = []
    for col in range(cols):
        total_score = 0
        for row in range(length_image):
            item = image[row][col][0]
            total_score += item
        score_to_index.append({"total_score": total_score, "index": col})
    score_to_index.sort(key = lambda x:x['total_score'])
    print(score_to_index)

    sorted_image = []
    for dict_item in score_to_index:
        index = dict_item['index']
        for row in range(length_image):
            item = image[row][index]
            if len(sorted_image) - 1 < row:
                sorted_image.append([])
            sorted_image[row].append(item)
    return sorted_image

def construct_image(image, cols):
    """ Constructs an image """

    array = np.arange(0, len(image) * cols, 1, np.uint8)
    array = np.reshape(array, (len(image), cols))
    np_array = np.array(image)
    # Find min and max values
    min = np_array.min()
    max = np_array.max()

    # Sort the image
    image_sorted = image
    if cols != 1:
        image_sorted = sort_image(image, cols)
    # Factor to normalize
    for x in range(len(image)):
        for y in range(cols):
            score = image_sorted[x]
            if cols != 1:
                score = image_sorted[x][y]
            colour = 255 * (score - min) / (max - min)
            array[x][y] = np.uint8(colour)


    data = im.fromarray(array)

    # saving the final output
    # as a PNG file
    data.save('image_features.png')
    return np_array

def predict_item_all(gmm, test_items):
    reshaped_test_items = []
    for test_item in test_items:
        reshaped_test_items.append(test_item.reshape(1, -1))
    total_score = 0
    logged_scores = gmm.score_samples(reshaped_test_items)
    for i in len(logged_scores):
        total_score += logged_scores[i]
    return total_score

def predict_item(gmm, test_item):
    """ Predicst the anomaly score for an item """
    test_item = test_item.reshape(1, -1)
    return gmm.score_samples(test_item)

def plot_scores(scores, labels):
    """ Plots the anomaly scores """
    scores = inverse_scores(scores)
    print(scores)
    print(labels)
    labels_hour = change_labels_to_hour(labels)
    threshold = get_threshold_normal(scores, labels)
    plt.plot(labels, scores)
    plt.axhline(y=threshold, color='r', linestyle=':', label='threshold')
    plt.axvline(x=datetime.strptime("08-30-20 8:21:00", "%m-%d-%y %H:%M:%S").strftime("%m-%d-%y %H:%M:%S"), color='r', linestyle='-', label='predicted time breach')
    plt.axvline(x=datetime.strptime("08-30-20 10:04:00", "%m-%d-%y %H:%M:%S").strftime("%m-%d-%y %H:%M:%S"), color='b', linestyle='-', label='expected time breach')

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.xlabel("Time (UTC) on day 30-08-2020")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.show()

def get_threshold_normal(scores, labels):
    anomaly = datetime.strptime("08-30-20 7:30:00", "%m-%d-%y %H:%M:%S")
    threshold = float('-inf')
    count = 0
    for label in labels:
        score = scores[count]
        if label < anomaly and threshold < score[0]:
            threshold = score
        count += 1
    return threshold

def change_labels_to_hour(labels):
    labels_hour = []
    for label in labels:
        labels_hour.append(label.strftime("%H:%M"))
    return labels_hour

def inverse_scores(scores):
    """ Inverses the scores for the plotting"""
    scaler = MinMaxScaler().fit(scores)
    total = scaler.data_max_ + scaler.data_min_
    for score in scores:
        score[0] = total - score[0]
    scores = turn_to_positive(scores)
    return scores

def turn_to_positive(scores):
    scaler = MinMaxScaler().fit(scores)
    max = scaler.data_max_
    min = scaler.data_min_
    if min >= 0:
        return scores
    for score in scores:
        score[0] = score[0] - min
    return scores

def main():
    """ Main method that gets the gaussian models and test data to predict the anomaly scores """
    gaussian_model = load_gaussian_models(GAUSSIAN_MODELS)
    test_data, labels, nodes_list = load_test_data(TEST_DATA)
    predict(gaussian_model, test_data, labels)


if __name__ == "__main__":
    main()
