# Install pytorch with pip with pip install torch torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch # install with pip install future
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import networkx as nx
import seaborn as sns
import statistics
import math
import matplotlib.dates as mdates
from datetime import datetime

NUM_FEATURES = 2384# wide: 2544, nz: 2384, serbia: 1656
NUM_HIDDEN = 1000
NUM_HIDDEN_2 = 600
CODE_FEATURES = int(1 + math.sqrt(NUM_FEATURES))
EPOCHS = 100
# MODEL_PATH = 'models/wide_model_normalized_closeness_entire_network.pth'
# NORMAL_SET = "wide/features_closeness_training_all.csv"
# ABNORMAL_SET = "wide/features_closeness_test.csv"
# MODEL_PATH = 'models/wide_model_normalized_degree_entire_network.pth'
# NORMAL_SET = "wide/wide_features_degree_normal_entire_network.csv"
# ABNORMAL_SET = "wide/wide_features_degree_abnormal_entire_network_day_30.csv"

# New Zealand Core Router
# MODEL_PATH = 'models/wide_degree.pth'
# #wide
# NORMAL_SET = "wide/wide_features_degree_normal_entire_network.csv"
# ABNORMAL_SET = "wide/features_degree_test.csv"

# serbia
# MODEL_PATH = 'soxrs/soxrs_model_normalized_closeness_entire_network.pth'
# NORMAL_SET = "soxrs/soxrs_features_closeness_normal_entire_network.csv"
# ABNORMAL_SET = "soxrs/soxrs_features_closeness_abnormal_entire_network_day_30.csv"

# NZ
MODEL_PATH = 'nz/model_normalized_degree_entire_network.pth'
NORMAL_SET = "nz/nz_features_degree_normal_entire_network.csv"
ABNORMAL_SET = "nz/nz_features_degree_entire_network_test.csv"
BATCH_SIZE = 80

class AE(nn.Module):
    """ Autoencoder class """
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.threshold = 0
        self.means = []
        self.standard_deviations = []
        self.layer_1 = nn.Linear(
            in_features = NUM_FEATURES, out_features = NUM_HIDDEN
        )
        self.layer_2 = nn.Linear(
            in_features = NUM_HIDDEN, out_features = CODE_FEATURES
        )
        self.layer_3 = nn.Linear(
            in_features = CODE_FEATURES, out_features = NUM_HIDDEN
        )
        self.layer_4 = nn.Linear(
            in_features = NUM_HIDDEN, out_features = NUM_FEATURES
        )

    def forward(self, features):
        """ Forward pass """
        # Encoder
        # Layer 1
        input_1 = self.layer_1(features)
        input_1 = torch.relu(input_1)

        # Layer 2
        hidden_1 = self.layer_2(input_1)
        hidden_1 = torch.relu(hidden_1)

        # Code layer
        hidden_2 = self.layer_3(hidden_1)
        hidden_2 = torch.relu(hidden_2)

        # Decoder
        # Layer 4
        output_1 = self.layer_4(hidden_2)

        return output_1

class DatasetClass(torch.utils.data.Dataset):
    """ Dataset class """
    def __init__(self, data_file, normalize):
        self.data = pd.read_csv(data_file, delimiter=",", usecols=range(1, NUM_FEATURES + 1), skiprows=0, dtype=np.float32)
        self.labels = pd.read_csv(data_file, delimiter=",", usecols=[0], dtype=np.unicode_)
        self.data_sorted = []
        self.labels_sorted = []
        self.means = []
        self.stdevs = []
        self.sort_data_by_date()
        self.normalize(normalize)
        self.X_train = torch.tensor(self.data_sorted, dtype=torch.float32)

    def normalize(self, normalize):
        """ Normalizes the values. If normalize is true then we normalize and compute the means and standard deviation """
        if not normalize:
            return
        # Iterating through the data and finding the means and standard deviation
        length_datas = len(self.data_sorted)
        for feature in range(NUM_FEATURES):
            features = []
            for data in range(length_datas):
                current_feature = self.data_sorted[data][feature]
                features.append(current_feature)
            mean = self.mean(features)
            standard_deviation = self.stdev(features)
            self.means.append(mean)
            self.stdevs.append(standard_deviation)
            # Normalizing the values
            for data in range(length_datas):
                if standard_deviation == 0.0:
                    self.data_sorted[data][feature] = 0.0
                else:
                    self.data_sorted[data][feature] = (self.data_sorted[data][feature] - mean) / standard_deviation

    def normalize_test(self, model):
        """ Normalize the values accordinging to the mean and standard deviation in the model """
        means = model.means
        stdevs = model.standard_deviations
        length_datas = len(self.data_sorted)
        for feature in range(NUM_FEATURES):
            mean = means[feature]
            standard_deviation = stdevs[feature]
            features = []

            for data in range(length_datas):
                if standard_deviation == 0.0:
                    # Avoid division by 0
                    self.data_sorted[data][feature] = 0
                else:
                    self.data_sorted[data][feature] = (self.data_sorted[data][feature] - mean) / standard_deviation
        self.X_train = torch.tensor(self.data_sorted, dtype=torch.float32)


    def mean(self, data):
        """ Finds the mean in a set of data """
        n = len(data)
        mean = sum(data) / n
        return mean

    def stdev(self, data):
        """ Finds the standard deviation """
        var = self.variance(data)
        std_dev = math.sqrt(var)
        return std_dev

    def variance(self, data):
        """ Finds the variance in the data """
        n = len(data)
        mean = sum(data) / n
        deviations = [(x - mean) ** 2 for x in data]
        variance = sum(deviations) / n
        return variance

    def sort_data_by_date(self):
        """ Sorts the data by date """
        labels_values = self.labels.values
        data_values = self.data.values
        sorted_data = []
        for i in range(len(labels_values)):
            date_string = labels_values[i][0][2:19]
            data = data_values[i]
            datetime_object = datetime.strptime(date_string, '%m-%d-%y %H:%M:%S')
            sorted_data.append({'date': datetime_object, 'data': data})
        sorted_data.sort(key = lambda x:x['date'])
        for i in range(len(sorted_data)):
            self.labels_sorted.append(sorted_data[i]['date'])
            self.data_sorted.append(sorted_data[i]['data'])

    def __len__(self):
        """ Gets the length of the data """
        return len(self.data)

    def __getitem__(self, index):
        """ Gets the data """
        return self.X_train[index]

def load_data(data_file, normalize):
    """ Loads the data files """
    data_x = DatasetClass(data_file, normalize)
    if "_normal" in data_file or "training" in data_file:
        global normal_dataset
        normal_dataset = data_x

    loader = torch.utils.data.DataLoader(
        data_x, batch_size=BATCH_SIZE, shuffle=normalize
    )
    return loader

def load_dataset(model):
    """ Loads the data sets """
    train_loader = load_data(NORMAL_SET, True)

    model.means = train_loader.dataset.means
    model.standard_deviations = train_loader.dataset.stdevs

    test_loader = load_data(ABNORMAL_SET, False)
    test_loader.dataset.normalize_test(model)
    return train_loader, test_loader
def instantiate_objects():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating a model
    model = AE().to(device) #input_shape=NUM_INSTANCES

    # Creating the optimizer
    optimizer = optim.Adam(model.parameters())

    # Using a mean squared error for loss
    criterion = nn.MSELoss()
    return device, model, optimizer, criterion

def predict(model, train_loader, device, istrain):
    """ Finds the error term for a loaded dataset """
    predictions, losses = [], []
    criterion = nn.MSELoss().to(device) # or use L1Loss

    with torch.no_grad(): # No need to calculate gradient as back prop is not used
        model.eval() # Evaluates the model
        for batch in train_loader:
            for instance in batch:
                batch_features = instance
                seq_true = batch_features.to(device)

                # Pass the input into the model
                seq_pred = model(seq_true)

                # Calculate the error
                loss = criterion(seq_pred, seq_true)

                # Append to the losses
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
                if istrain and loss.item() > model.threshold:
                    model.threshold = loss.item()
                print("loss: " + str(loss.item()))
    return predictions, losses

def test(model, test_loader, device):
    """ Finds the loss of the test set """
    print("\n--Testing--")
    predictions, pred_losses = predict(model, test_loader, device, False)

    # If the errror is greater than the threshold, it is classified as anomaly
    correct = sum(loss > model.threshold for loss in pred_losses)
    correct_predictions = (correct + 0.0) / len(test_loader)
    print('Correct anomaly predictions: {:.3f}'.format(correct_predictions))
    print("threshold: " + str(model.threshold))
    # Plotting
    plt.plot(test_loader.dataset.labels_sorted, pred_losses)
    print(test_loader.dataset.labels_sorted)
    # plt.axhline(y=model.threshold, color='r', linestyle=':', label='threshold')
    plt.axhline(y=0.11, color='r', linestyle=':', label='threshold')
    plt.axvline(x=datetime.strptime("08-30-20 10:19:00", "%m-%d-%y %H:%M:%S").strftime("%m-%d-%y %H:%M:%S"), color='b', linestyle='-', label='expected time breach')
    plt.axvline(x=datetime.strptime("08-30-20 9:28:01", "%m-%d-%y %H:%M:%S").strftime("%m-%d-%y %H:%M:%S"), color='r', linestyle='-', label='predicted time breach')
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.xlabel("Time (UTC) on day 30-08-2020")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.show()

def predict_threshold(model, train_loader, device):
    """ Predicts the threshold using the error rate of the normal data set """
    print("\n--Predicting Threshold--")
    _, losses = predict(model, train_loader, device, True)

    # Visualize the loss
    plt.plot(normal_dataset.labels_sorted, losses)
    # Plotting Threshold
    plt.axhline(y=model.threshold, color='r', linestyle='-', label='threshold')
    print(len(train_loader))
    plt.xlabel("Time (month-day hour)")
    plt.ylabel("Error Rate")
    plt.show()
    print("threshold: " + str(model.threshold))

# Main Functions
def main():
    """
    Loads the dataset and trains the model.
    Prediction of abnormal and normal data is conducted
    """
    device, model, optimizer, criterion = instantiate_objects()
    train_loader, test_loader = load_dataset(model)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        loss = 0
        for batch in train_loader:
            batch_features = batch
            # Reset the gradients to 0
            optimizer.zero_grad()

            # Compute reconstructions
            outputs = model(batch_features)

            # Compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # Compute accumlated gradients
            train_loss.backward()

            # Perform parameter update
            optimizer.step()

            # Add mini-batch training loss to epoch loss
            loss += train_loss.item()
        # Compute epoch training loss
        loss = loss / len(train_loader)
        losses.append(loss)

        # Display epoch training process
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
    # Displays the error
    x = np.linspace(0, EPOCHS-1, EPOCHS)
    print(x)
    plt.plot(x, losses, label="Error Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # Predicting the threshold
    predict_threshold(model, train_loader, device)
    #
    # Saves the model
    torch.save(model, MODEL_PATH)

    # Predicting the test set
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
