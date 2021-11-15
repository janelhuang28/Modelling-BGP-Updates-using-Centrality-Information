from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import pandas as pd
# wide
NORMAL_DATA = "nz/nz_features_degree_normal_entire_network.csv"
# NORMAL_DATA = "wide/features_degree_training_all.csv"
# serbia
# NORMAL_DATA = "soxrs/soxrs_features_closeness_normal_entire_network.csv"
NUM_FEATURES = 2384 #NZ: 2385, NZ (2526), Wide (closeness: 2544, degree: 2544): remove (47872, 265066, 8121)
def load_data(fileName):
    """ Loads the data by column and find the gaussian mixture model for each column """
    gaussian_models = []

    # Get the column data
    # data_x = np.loadtxt(fileName, dtype=np.float32, delimiter=',', usecols=(1, NUM_FEATURES), skiprows=1, unpack=True)
    data_x =  pd.read_csv(fileName, delimiter=",", usecols=range(1, NUM_FEATURES + 1), skiprows=0, dtype=np.float32)
    # data_x.data
    # print(data_x.data)
    gm = GaussianMixture(n_components=1).fit(data_x)
    num_rows, num_cols = data_x.shape
    print(data_x.shape)
    with open("nz/gaussian_full_models_degree_1.pkl", "wb") as fp:
        pickle.dump(gm, fp)
        fp.close()




def show_model(data, gaussian_model):
    """ Shows the model """
    # xpdf = np.linspace(-2, 2, 100)
    # density = np.exp(clf.score(xpdf))
    #
    # plt.hist(data, 80, normed=True, alpha=0.5)
    # plt.plot(xpdf, density, '-r')
    # plt.xlim(-2, 2)
    # Bell Curve
    # mean = gaussian_model.means_
    # standard_deviation =np.sqrt(gmm.covariances_)
    # x_values = data
    # y_values = scipy.stats.norm(mean, standard_deviation)
    # plt.plot(x_values, y_values.pdf(x_values))
    plt.xlabel("Centrality")
    plt.ylabel("Number of Instances")
    plt.hist(data, color="lightblue")
    plt.show()

# Main Functions
def main():
    load_data(NORMAL_DATA)

if __name__ == "__main__":
    main()
