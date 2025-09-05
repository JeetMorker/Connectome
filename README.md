# TBI Connectome Visualization & Analysis

This project provides automated tools to process, visualize, and analyze structural brain connectomes in patients with traumatic brain injury (TBI) and healthy controls. The goal is to identify network-level biomarkers of TBI and explore how injury severity impacts brain connectivity.

## 🧠 Overview

- Practice and output folders show results of practice visualization of connectomes on the 1000Brains dataset
- TBI.py is the pipline used for actual research and results are shown in the png file

The project enables:

- Generation and analysis of structural connectomes  
- Computation of **Network Normality Scores (NNS)** to quantify connectome similarity  
- Graph theoretical analysis including clustering coefficient, global efficiency, centrality, small-worldness, and modularity  
- Visualization of network similarity, graph metrics, and correlations with cognitive outcomes (e.g., post-traumatic amnesia, memory, executive function)  

## 🔧 Features

- **Structural Connectomes:** Heatmaps, node strength, connectivity analysis  
- **Network Similarity:** Violin plots for patients vs. healthy controls  
- **Graph Metrics & Correlations:** Scatter plots, regression analysis with cognitive scores  
- **Statistical Analysis:** Mann-Whitney U, Wilcoxon signed-rank, effect sizes, linear regression  
- **Modular Python Code:** Helper functions for reproducible and scalable workflows  

