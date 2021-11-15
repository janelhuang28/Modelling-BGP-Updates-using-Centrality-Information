# Generating Graphs 

The main.py generates the graphs given a bview and an update file. The generated graph is in textual format where each line represents a given node (AS) and its neighbour. 

# Articulation Points

The articulation_points.py finds the articulation points given in a graph.

Each tuple that is used within the articulation points algorithm is stored in ap_tuple.py

# Features - Centrality

The main_feature.py file will process the generated graphs and calculate the feature values.

The centrality value will be calculated via centrality.py. This will calculate both the closeness and the degree centrality.

# Autoencoders

To train the autoencoder and generate a model, Autoencoders.py will be used that takes a csv file and parses the features within it. A model will be saved in the file specified in the MODEL_PATH

To find the predicted output of the features, Autoencoders_predict.py will be used. This takes in a csv file and predicts the output of each datapoint in the test set.
