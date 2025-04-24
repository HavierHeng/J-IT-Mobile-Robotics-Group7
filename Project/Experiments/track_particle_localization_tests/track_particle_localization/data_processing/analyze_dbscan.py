import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns

# Load data
all_obs_df = pd.read_csv('all_observations.csv')
static_obs_df = pd.read_csv('objects.csv')

# Parameters to test
eps_values = [1.0, 2.0, 3.0]  # cluster_distance
min_samples_values = [3, 5, 7]  # min_observations

# Global Cluster Plot
plt.figure(figsize=(10, 8))
for eps in eps_values:
    for min_samples in min_samples_values:
        # Group by class
        classes = all_obs_df['Class'].unique()
        plt.clf()
        plt.title(f'Global DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)

        for cls in classes:
            cls_data = all_obs_df[all_obs_df['Class'] == cls]
            positions = cls_data[['X', 'Y']].values

            # Run DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
            labels = clustering.labels_

            # Plot all observations
            sns.scatterplot(
                x=cls_data['X'], y=cls_data['Y'],
                label=f'{cls} (raw)', alpha=0.5, s=50
            )

            # Plot clusters
            unique_labels = set(labels) - {-1}
            for label in unique_labels:
                mask = labels == label
                plt.scatter(
                    positions[mask, 0], positions[mask, 1],
                    label=f'{cls} Cluster {label}', s=100, marker='o'
                )

            # Plot noise
            noise_mask = labels == -1
            if np.any(noise_mask):
                plt.scatter(
                    positions[noise_mask, 0], positions[noise_mask, 1],
                    c='gray', marker='x', s=50, label='Noise'
                )

        # Plot static objects
        for _, row in static_obs_df.iterrows():
            plt.scatter(
                row['X'], row['Y'], c='black', marker='*', s=200,
                label='Static Object' if _ == 0 else None
            )

        plt.legend()
        plt.savefig(f'global_dbscan_eps{eps}_min{min_samples}.png')
        plt.clf()

# Sanity Check: Local Clustering Plot
plt.figure(figsize=(10, 8))
# Group by timestep (approximate batches by rounding timestamps to 0.1s)
all_obs_df['Timestep'] = all_obs_df['Timestamp'].round(1)
timesteps = all_obs_df['Timestep'].unique()
n_timesteps = len(timesteps)

# Plot a subset of timesteps to avoid clutter
max_plots = 16
timesteps_to_plot = timesteps[:max_plots]
rows = int(np.ceil(np.sqrt(len(timesteps_to_plot))))
cols = int(np.ceil(len(timesteps_to_plot) / rows))

for i, timestep in enumerate(timesteps_to_plot):
    plt.subplot(rows, cols, i+1)
    data = all_obs_df[all_obs_df['Timestep'] == timestep]
    positions = data[['X', 'Y']].values

    # Run DBSCAN (use current parameters: eps=2.0, min_samples=5)
    clustering = DBSCAN(eps=2.0, min_samples=5).fit(positions)
    labels = clustering.labels_

    # Plot observations
    for cls in data['Class'].unique():
        cls_data = data[data['Class'] == cls]
        plt.scatter(
            cls_data['X'], cls_data['Y'],
            label=cls, alpha=0.5, s=50
        )

    # Plot clusters
    unique_labels = set(labels) - {-1}
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            positions[mask, 0], positions[mask, 1],
            label=f'Cluster {label}', s=100, marker='o'
        )

    # Plot noise
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(
            positions[noise_mask, 0], positions[noise_mask, 1],
            c='gray', marker='x', s=50, label='Noise'
        )

    plt.title(f'Timestep {timestep:.1f}s')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
# plt.savefig('local_dbscan_sanity_check.png')
