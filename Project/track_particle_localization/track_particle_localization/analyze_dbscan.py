#!/usr/bin/env python3

"""
Data visualization script to help us bruteforce to find the best DBScan parameter offline
- Since our clustering affects how we summarize the final CSV
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

def load_observations(csv_file, class_filter='person', time_window=10.0):
    """Load and filter observations from all_observations.csv."""
    df = pd.read_csv(csv_file)
    df['Timestamp'] = df['Timestamp'].astype(float)
    max_time = df['Timestamp'].max()
    df = df[(df['Timestamp'] >= max_time - time_window) & (df['Class'] == class_filter)]
    return df

def plot_clusters(df, eps=2.0, min_samples=5, output_file='global_dbscan.png'):
    """Apply DBSCAN and plot clusters."""
    positions = df[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df['X'], y=df['Y'], hue=labels, palette='deep', style=(labels == -1), size=(labels == -1))
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = positions[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        plt.scatter(centroid[0], centroid[1], c='red', marker='x', s=200, label='Centroids' if label == min(set(labels)) else None)
    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

    # Print stats
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    for label in set(labels):
        if label == -1:
            continue
        cluster_df = df[labels == label]
        print(f"Cluster {label}: {len(cluster_df)} points, Avg Variance: {cluster_df['Variance'].mean():.4f}")

def main():
    csv_file = 'all_observations.csv'
    class_filter = 'person'  # Adjust based on your object classes
    time_window = 10.0  # Match track_localization.py's time_window
    eps = 2.0  # Match cluster_distance
    min_samples = 5  # Match min_observations

    df = load_observations(csv_file, class_filter, time_window)
    if df.empty:
        print("No observations found for class", class_filter)
        return
    output_file = f'global_dbscan_eps{eps}_min{min_samples}.png'
    plot_clusters(df, eps, min_samples, output_file)
    print(f"Plot saved as {output_file}")

if __name__ == '__main__':
    main()
