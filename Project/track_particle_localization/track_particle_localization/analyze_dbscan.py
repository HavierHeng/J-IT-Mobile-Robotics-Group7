#!/usr/bin/env python3

"""
Visualize all observations with DBSCAN clustering and color by object class.
Useful for visually tuning DBSCAN parameters offline.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def load_observations(csv_file, time_window=600.0, min_variance=0.001):
    """Load and filter recent observations from CSV, excluding high-variance points."""
    df = pd.read_csv(csv_file)
    df['Timestamp'] = df['Timestamp'].astype(float)
    max_time = df['Timestamp'].max()
    df = df[df['Timestamp'] >= max_time - time_window]
    
    if min_variance is not None:
        initial_count = len(df)
        df = df[df['Variance'] <= min_variance]
        filtered_count = initial_count - len(df)
        print(f"Filtered out {filtered_count} points with variance > {min_variance}")
    
    return df

def plot_clusters(df, eps=0.5, min_samples=2, output_file='global_dbscan.png', show=False):
    """Apply DBSCAN and plot observations colored by class, with centroids and all variances."""
    positions = df[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_

    df['Cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Prepare data for objects.csv-like output
    objects_data = []

    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='X', y='Y', hue='Class', style='Cluster', palette='tab10', s=70)

    for label in set(labels):
        if label == -1:
            continue

        cluster_df = df[df['Cluster'] == label]
        min_var = cluster_df['Variance'].min()

        cluster_points = cluster_df[['X', 'Y']].values
        centroid = np.mean(cluster_points, axis=0)
        
        # Get the point closest to the centroid within the cluster
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx = cluster_df.index[distances.argmin()]
        closest_point = cluster_df.loc[closest_idx]

        # Plot it with a distinct marker
        plt.scatter(closest_point['X'], closest_point['Y'],
                    c='blue', marker='D', s=180,
                    label='Closest to Centroid' if label == min(set(labels)) else None)

        # Annotate its variance
        plt.text(closest_point['X'] + 0.1, closest_point['Y'] - 0.2,
                 f"var: {closest_point['Variance']:.4f}",
                 fontsize=9, color='blue', backgroundcolor='white')

        # Red X for centroid
        plt.scatter(centroid[0], centroid[1], c='red', marker='x', s=200,
                    label='Centroid' if label == min(set(labels)) else None)

        # Min variance label at centroid
        plt.text(centroid[0] + 0.1, centroid[1] + 0.1,
                 f"min var: {min_var:.4f}",
                 fontsize=10, color='black', backgroundcolor='white')

        # Star marker for the min variance point
        min_var_idx = cluster_df['Variance'].idxmin()
        min_var_point = cluster_df.loc[min_var_idx]
        plt.scatter(min_var_point['X'], min_var_point['Y'],
                    c='black', marker='*', s=250,
                    label='Min Variance Point' if label == min(set(labels)) else None)

        # Add variance labels for all points in the cluster
        for idx, row in cluster_df.iterrows():
            plt.text(row['X'] + 0.05, row['Y'] + 0.05,
                     f"{row['Variance']:.4f}",
                     fontsize=8, color='black')

        # Store data for objects.csv-like output
        objects_data.append({
            'Cluster_ID': label,
            'Class': cluster_df['Class'].iloc[0],  # Assume all points in cluster have same class
            'X': centroid[0],
            'Y': centroid[1],
            'Z': closest_point['Z'],
            'Variance': closest_point['Variance']
        })

    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()

    plt.xlim(left=0)

    if show:
        plt.show()
    else:
        plt.savefig(output_file)
        print(f"Plot saved as {output_file}")
    plt.close()

    # Print stats
    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    for label in set(labels):
        if label == -1:
            continue
        cluster_df = df[df['Cluster'] == label]
        min_var = cluster_df['Variance'].min()
        centroid = np.mean(cluster_df[['X', 'Y']].values, axis=0)
        print(f"Cluster {label}: {len(cluster_df)} points, Avg Var: {cluster_df['Variance'].mean():.4f}, Min Var: {min_var:.4f}, Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f})")

    # Save objects.csv-like output
    with open('dbscan_objects.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cluster_ID', 'Class', 'X', 'Y', 'Z', 'Variance'])
        for obj in objects_data:
            writer.writerow([
                obj['Cluster_ID'],
                obj['Class'],
                obj['X'],
                obj['Y'],
                obj['Z'],
                obj['Variance']
            ])
    print("Saved cluster data to dbscan_objects.csv")

def main():
    parser = argparse.ArgumentParser(description="Visualize DBSCAN clustering across all object classes.")
    parser.add_argument('csv_file', help='Path to the observations CSV file.')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN epsilon parameter (default: 0.5)')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter (default: 2)')
    parser.add_argument('--time_window', type=float, default=600.0, help='Time window in seconds (default: 600.0)')
    parser.add_argument('--min_variance', type=float, default=0.001,
                        help='Exclude points with variance above this threshold before clustering (default: 0.001)')
    parser.add_argument('--show', action='store_true', help='Show plot instead of saving to file')
    args = parser.parse_args()

    df = load_observations(args.csv_file, args.time_window, args.min_variance)
    if df.empty:
        print("No observations found after filtering.")
        return

    output_file = f'global_dbscan_eps{args.eps}_min{args.min_samples}.png'
    plot_clusters(df, args.eps, args.min_samples, output_file=output_file, show=args.show)

if __name__ == '__main__':
    main()
