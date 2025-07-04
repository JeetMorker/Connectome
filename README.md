# Connectome Visualization & Analysis (1000BRAINS Dataset)

This project focuses on the automated processing, visualization, and exploratory analysis of structural and functional brain connectomes from the 1000BRAINS dataset. The goal is to enable scalable and reproducible workflows for neuroimaging research using modular Python code.

## 🧠 Overview

Using a set of scan IDs (e.g., 0001_1, 0001_2, 0002_1, 0002_2), the project:
- Generates structural and functional connectomes
- Computes and visualizes metrics such as node strength and edge distributions
- Analyzes demographic and cognitive data
- Creates a variety of visualizations (heatmaps, histograms, scatter plots, box/violin plots)

## 🔧 Features

- Structural Connectomes: Heatmaps and node strength analysis
- Functional Connectomes: Full, positive, and negative edge visualization
- Histograms: Edge weight distributions, age, brain volume, cognitive scores
- Box & Violin Plots: Node strengths, demographics, and intelligence metrics
- Scatter Plots: Structural vs functional edges, age vs volume, cognitive comparisons
- Modular Design: Helper functions for clean, reusable code

## 🚀 Usage

1. Update file paths in visualize.py to match your local dataset layout:
   struct_base_dir = '/path/to/structural/connectomes'
   func_base_dir = '/path/to/functional/connectomes'
   demo_csv = '/path/to/1000Brains_demographics_volume.csv'
   scores_csv = '/path/to/1000BRAINS_scores.csv'

3. Run the main script  
   python visualize.py

4. Find outputs in the output/ directory.

## 🧰 Technologies Used

- numpy – Matrix operations, node strength, thresholds
- pandas – CSV handling, demographic and score parsing
- matplotlib – Plotting (heatmaps, histograms, scatter, etc.)
- seaborn – Advanced box and violin plots (with jittered scatter)

## 💡 Notes

- Diagonal entries in connectomes are zeroed to ignore self-connections.
- Thresholding and masking (e.g., for negative FC values) are configurable.
- Designed to support scaling up to the full 1000BRAINS dataset.

