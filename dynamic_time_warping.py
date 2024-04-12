from fastdtw import fastdtw
import numpy as np
import pandas as pd
from tslearn.barycenters import dtw_barycenter_averaging
import os

from preprocessing import UCRDataset,  PrematureDatasetSplit

def calculate_distance_target(source_dataset_path, targetset):
    # Get source data set
    source_dataset = UCRDataset(source_dataset_path)

    # Calculate barycenter average of source data set
    source_average = dtw_barycenter_averaging(source_dataset.X, max_iter = 5)

    tot_dist = 0
    for i in range(targetset.X.shape[-1]):
        # Calculate barycenter average of target dataset and DTW distance to source data set
        avg = dtw_barycenter_averaging(targetset.X[:, :, i], max_iter = 5)
        distance, path = fastdtw(avg, source_average, dist = 2) # Equal to Euclidian distance
        
        tot_dist += distance
        
        print(f"Distance of source data set and target feature {i}: {distance}")

    return tot_dist

def loop_dtw_calculation(source_data_directory):
    results = {}
    data = pd.read_csv("./data/train_test_data/trainval.csv", index_col=0)
    targetset = PrematureDatasetSplit(data)
    
    for folder_name in os.listdir(source_data_directory):
        folder_path = os.path.join(source_data_directory, folder_name)
        if os.path.isdir(folder_path):  
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.tsv') and 'TRAIN' in file_name:
                    print(f"Filename: {file_name}:")
                    file_path = os.path.join(folder_path, file_name)
                    
                    distance = calculate_distance_target(file_path, targetset)
                    print(f"Total distance: {distance}")
                    
                    results[file_name] = distance
    file_name = "./results/dtw_results.txt"
                    
    with open(file_name, "w") as f:
        for key, value in results.items():
            f.write("{}: {} \n".format(key, value))

    