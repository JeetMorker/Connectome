import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_connectome(matrix, output_file='connectome.png', cmap='viridis'):
    """
    Plot and save a connectome heatmap.
    """
    matrix = np.array(matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap=cmap, square=True, center=0 if 'coolwarm' in cmap else None, cbar=True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency', output_file='histogram.png'):
    """
    Plot and save a histogram of numerical data.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_scatter(x, y, output_file='scatter.png', title='Scatter Plot', xlabel='X', ylabel='Y',
                 color='blue', label=None, fit_curve=False, curve_degree=1):
    """
    Generate and save a scatter plot with optional polynomial curve fitting.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, c=color, label=label, alpha=0.6, edgecolors='w', linewidth=0.5)

    if fit_curve:
        coeffs = np.polyfit(x, y, deg=curve_degree)
        poly_eq = np.poly1d(coeffs)
        xs = np.linspace(min(x), max(x), 1000)
        plt.plot(xs, poly_eq(xs), color='red', linewidth=2, label=f'Poly fit deg={curve_degree}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label is not None or fit_curve:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_boxplot(data_arrays, labels, title='Boxplot', ylabel='Value', output_file='boxplot.png',
                 show_scatter=True, figsize=(10, 6), colors=None):

    plt.figure(figsize=figsize)
    boxprops = dict(linewidth=2)
    medianprops = dict(linewidth=2, color='firebrick')
    
    bp = plt.boxplot(data_arrays, labels=labels, patch_artist=True,
                     boxprops=boxprops, medianprops=medianprops)

    if colors:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    if show_scatter:
        for i, data in enumerate(data_arrays):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            plt.scatter(x, data, alpha=0.5, s=10, color='black')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_violin(data_arrays, labels, title='Violin Plot', ylabel='Value', output_file='violin.png',
                show_scatter=True, figsize=(10, 6), colors=None):
    """
    Generate and save a violin plot for multiple data arrays.

    Parameters:
    - data_arrays: list of 1D numeric arrays
    - labels: list of group labels
    - title, ylabel, output_file: plot metadata
    - show_scatter: overlay individual points
    - figsize: tuple size
    - colors: optional list of colors
    """

    # Flatten to long format
    data_flat = []
    group_flat = []
    for data, label in zip(data_arrays, labels):
        data_flat.extend(data)
        group_flat.extend([label] * len(data))

    df = pd.DataFrame({'Value': data_flat, 'Group': group_flat})

    plt.figure(figsize=figsize)
    sns.violinplot(x='Group', y='Value', data=df, inner='box', palette=colors)
    
    if show_scatter:
        sns.stripplot(x='Group', y='Value', data=df, color='black', size=2.5, alpha=0.4, jitter=True)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

