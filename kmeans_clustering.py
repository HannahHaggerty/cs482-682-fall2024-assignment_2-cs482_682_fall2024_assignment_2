import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None

        self.dataset_file = self.get_dataset_file(dataset_file)
        self.read_mat()

    #Read the dataset using scipy.io.loadmat
    def get_dataset_file(self, dataset_file):
        # Map numeric inputs to actual file names
        if dataset_file == '1':
            return 'dataset_q2.mat'
        else:
            return dataset_file

    def read_mat(self):
        try:
            mat = scipy.io.loadmat(self.dataset_file)
            self.data = mat['X']  #Adjust the key
        except FileNotFoundError: #handle errors
            print(f"Error: The file {self.dataset_file} was not found.")
            exit(1)

    def model_fit(self):
        '''
        Initialize KMeans using sklearn and execute K-means clustering
        '''
    
        self.model = KMeans(n_clusters=3, max_iter=300, random_state=0) #K means clustering find 3 clusters running a maximum of 300 iterations w/ fixed random state 
    
        # Fit the model to the data
        self.model.fit(self.data)
    
        
        cluster_centers = self.model.cluster_centers_ #get the the cluster centers
        return cluster_centers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d', '--dataset_file', type=str, default="dataset_q2.mat", help='path to dataset file or a number like 1 or 2')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    cluster_centers = classifier.model_fit()
    print("Cluster Centers:\n", cluster_centers)
