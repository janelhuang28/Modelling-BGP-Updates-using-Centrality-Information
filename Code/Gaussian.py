from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
# wide
NORMAL_DATA = "nz/nz_features_closeness_normal_entire_network.csv"
# NORMAL_DATA = "wide/features_closeness_training_all.csv"
# serbia
# NORMAL_DATA = "soxrs_soxrs/soxrs_features_closeness_normal_entire_network_next_days.csv"
# NORMAL_DATA = "soxrs_soxrs/soxrs_features_degree_normal_entire_network_next_days.csv"
NUM_FEATURES = 2384 #NZ: 2385, NZ (2526), Wide (closeness: 2544, degree: 2544): remove (47872, 265066, 8121)
def load_data(fileName):
    """ Loads the data by column and find the gaussian mixture model for each column """
    gaussian_models = []
    for i in range(1, NUM_FEATURES + 1):
        # Get the column data
        data_x = np.loadtxt(fileName, dtype=np.float32, delimiter=',', usecols=(i), skiprows=1, unpack=True)

        # Reshape as the data is only a 1D array
        data = data_x.reshape(-1, 1)

        # Generate a gmm with 3 maximum components
        gm = GaussianMixture(n_components=1).fit(data)

        # Add to the list of gaussian models
        gaussian_models.append(gm)
        print(i)
        # show_model(data, gm)
    with open("nz/gaussian_models_closeness_1.pkl", "wb") as fp:
        pickle.dump(gaussian_models, fp)
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
