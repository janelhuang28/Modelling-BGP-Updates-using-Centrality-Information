import torch
import torch.nn as nn
from Autoencoders import AE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from Autoencoders import DatasetClass

MODEL_PATH = "models/model_normalized_degree_entire_network.pth"
TEST_SET = "features_degree_test_entire_network.csv"
NUM_FEATURES = 2547

def predict(model, train_loader, device):
    """ Finds the error term for a loaded dataset """
    predictions, losses = [], []
    criterion = nn.MSELoss().to(device) # or use L1Loss
    with torch.no_grad(): # No need to calculate gradient as back prop is not used
        model.eval() # Evaluates the model
        for batch in train_loader:
            print(batch[0])
            batch_features = batch[0]
            seq_true = batch_features.to(device)

            # Pass the input into the model
            seq_pred = model(seq_true)

            # Calculate the error
            loss = criterion(seq_pred, seq_true)

            # Append to the losses
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
            print(loss.item())
    return predictions, losses

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
    predictions, pred_losses = predict(model, test_loader, device)
    print(len(pred_losses))
    print(pred_losses)
    # If the errror is greater than the threshold, it is classified as anomaly
    anomalies = sum(loss > model.threshold for loss in pred_losses)
    anomaly_rate = (anomalies + 0.0) / len(test_loader)
    print('Number of anomaly predictions: {:.3f}'.format(anomaly_rate))

    # If the errror is greater than the threshold, it is classified as anomaly
    normal = sum(loss <= model.threshold for loss in pred_losses)
    normal_rate = (normal + 0.0) / len(test_loader)
    print('Number of normal predictions: {:.3f}'.format(normal_rate))

    # Plotting the losses
    # sns.distplot(pred_losses, bins=50, kde=True);
    plt.plot(test_loader.dataset.labels_sorted, pred_losses)
    print(test_loader.dataset.labels_sorted)
    plt.axhline(y=model.threshold, color='r', linestyle=':', label='threshold')
    plt.xlabel("Time (month-dath hour)")
    plt.ylabel("Error Rate")
    plt.show()

def main():
    """ Loads the model and predicts the dataset """
    # Loads the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH)
    model = model.to(device)

    # Loads the test
    dataset_loader = load_data(TEST_SET, model)
    test(model, dataset_loader, device)

if __name__ == "__main__":
    main()
