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

MODEL_PATH = "nz/model_normalized_closeness_entire_network.pth"
TEST_SET = "nz/nz_features_closeness_entire_network_test.csv"
VALIDATION_SET = "wide/features_degree_validation_set.csv"
NUM_FEATURES = 2384

GAUSSIAN_MODELS = "nz/gaussian_models_closeness.pkl"
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

def predict(model, train_loader, device):
    """ Finds the error term for a loaded dataset """
    predictions, losses = [], []
    criterion = nn.MSELoss().to(device) # or use L1Loss
    anomaly = datetime.strptime("08-30-20 09:30:00", "%m-%d-%y %H:%M:%S")
    with torch.no_grad(): # No need to calculate gradient as back prop is not used
        model.eval() # Evaluates the model
        count = 0
        for batch in train_loader:
            batch_features = batch[0]
            seq_true = batch_features.to(device)

            # Pass the input into the model
            seq_pred = model(seq_true)

            # Calculate the error
            loss = criterion(seq_pred, seq_true)

            # Append to the losses
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
            if train_loader.dataset.labels_sorted[count] < anomaly and loss.item() > model.threshold:
                model.threshold = loss.item()
            count += 1
    return predictions, losses, model

def load_data(data_file, model):
    """ Loads the data files """
    data_x = DatasetClass(data_file, False)
    data_x.normalize_test(model)

    loader = torch.utils.data.DataLoader(
        data_x, batch_size=1, shuffle=False
    )
    return loader

def test(model, test_loader, device):
    """ Finds the loss of the test set """
    print("\n--Testing--")
    predictions, pred_losses, model = predict(model, test_loader, device)
    return pred_losses

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

def generate_scatter_plot(ae_scores, gmm_scores, labels):
    """
    Generates a scatter plot
    """
    g1 = (labels, ae_scores)
    gmm_scores = scale_gmm_scores(ae_scores, gmm_scores)
    g2 = (labels, gmm_scores)

    data = (g1, g2)
    colors = ("red", "green")
    groups = ("AE", "GMM")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('GMM vs Autoencoder')
    plt.xlabel("Time (UTC) on day 30-08-2020")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    """ Loads the model and predicts the dataset """
    # Autoencoders
    # Loads the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH)
    model = model.to(device)
    # Loads the test
    dataset_loader = load_data(TEST_SET, model)
    ae_scores = test(model, dataset_loader, device)

    # GMM
    gaussian_models = load_gaussian_models(GAUSSIAN_MODELS)
    test_data, labels, nodes_list = load_test_data(TEST_DATA)
    gmm_scores, labels = predict_gaussian(gaussian_models, test_data, labels)

    generate_scatter_plot(ae_scores, gmm_scores, labels)
if __name__ == "__main__":
    main()
