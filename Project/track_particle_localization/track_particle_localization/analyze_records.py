#!/usr/bin/env python3

"""
Visualize raw observations from all_observations.csv with points colored by class and variance annotated.
No DBSCAN clustering or filtering; just plots the raw data for inspection.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_observations(csv_file, time_window):
    """Load and filter recent observations from CSV."""
    df = pd.read_csv(csv_file)
    df['Timestamp'] = df['Timestamp'].astype(float)
    max_time = df['Timestamp'].max()
    df = df[df['Timestamp'] >= max_time - time_window]
    return df

def plot_observations(df, output_file='observations.png', time_window=600.0, show=False):
    """Plot observations colored by class with variance annotations."""
    if df.empty:
        print("No observations found in the given time window.")
        return

    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='X', y='Y', hue='Class', palette='tab10', s=70)

    # Add variance labels for all points
    for idx, row in df.iterrows():
        plt.text(row['X'] + 0.05, row['Y'] + 0.05,
                 f"{row['Variance']:.4f}",
                 fontsize=8, color='black')

    plt.title(f'Raw Observations (time_window={time_window}s)\nTotal Points: {len(df)}')
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

def main():
    parser = argparse.ArgumentParser(description="Visualize raw observations from all_observations.csv.")
    parser.add_argument('csv_file', help='Path to the observations CSV file (e.g., all_observations.csv).')
    parser.add_argument('--time_window', type=float, default=600.0, help='Time window in seconds (default: 600.0)')
    parser.add_argument('--show', action='store_true', help='Show plot instead of saving to file')
    args = parser.parse_args()

    df = load_observations(args.csv_file, args.time_window)
    if df.empty:
        print("No observations found in the given time window.")
        return

    output_file = 'observations.png'
    plot_observations(df, output_file=output_file, time_window=args.time_window, show=args.show)

if __name__ == '__main__':
    main()
