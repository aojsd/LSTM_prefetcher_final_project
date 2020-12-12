import argparse
import pandas as pd
import torch
from sklearn.cluster import KMeans

def address_kmeans(data):
    #clustering the data by the address
    kmeans6 = KMeans(n_clusters = 6)
    clusters = kmeans6.fit_predict(data[["pc"]])
    print(clusters)

    return clusters

# Load data from input file
def load_data(infile, nrows, skip=None, batch_size=2):
    # Same function is used to read training, validation, and test datasets, 
    # so we need to skip ahead in the file to avoid using the same data for
    # training and testing.
    if skip == None:
        data = pd.read_csv(infile, nrows=nrows)
    else:
        data = pd.read_csv(infile, nrows=nrows, skiprows=range(1, skip + 1))

    clusters = address_kmeans(data)
    data["cluster"] = clusters
    print(data.cluster.unique())

    print(data)
    

    # Convert data to PyTorch tensors
    pc = torch.tensor(data["pc"].to_numpy())
    delta_in = torch.tensor(data["delta_in"].to_numpy())
    types = torch.tensor(data["type"].to_numpy())
    targets = torch.tensor(data["delta_out"].to_numpy())
    clusters = torch.tensor(data["cluster"].to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, types, targets, clusters)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter

if __name__ == "__main__":

    load_data("data/15k_nostride.csv", nrows = 15000)
