import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import plot_connectome, plot_histogram, plot_scatter, plot_boxplot, plot_violin

# === Paths and IDs ===
struct_base_dir = '/Users/jeetmorker/Downloads/1000brains/070-DesikanKilliany/SC'
func_base_dir = '/Users/jeetmorker/Downloads/1000brains/070-DesikanKilliany/FC'
demo_file = '/Users/jeetmorker/Downloads/1000brains/1000Brains_demographics_volume.csv'
scores_file = '/Users/jeetmorker/Downloads/1000brains/1000BRAINS_scores.csv'

output_struct_dir = 'output/structural_connectomes'
output_func_dir = 'output/functional_connectomes'
output_hist_dir = 'output/histograms'
output_scatter_dir = 'output/scatterplots'

for d in [output_struct_dir, output_func_dir, output_hist_dir, output_scatter_dir]:
    os.makedirs(d, exist_ok=True)

scan_ids = ['0001_1', '0001_2', '0002_1', '0002_2']

def map_struct_filename(scan_id):
    return f'{scan_id}_Counts.csv'

def map_func_filename(scan_id):
    return f'{scan_id}_RestEmpCorrFC.csv'

def upper_triangular_values(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]


# === Structural connectomes ===
for scan_id in scan_ids:
    struct_path = os.path.join(struct_base_dir, map_struct_filename(scan_id))
    try:
        struct_mat = np.loadtxt(struct_path, delimiter=',')
    except Exception as e:
        print(f'Error loading structural {struct_path}: {e}')
        continue

    np.fill_diagonal(struct_mat, 0)
    output_path = os.path.join(output_struct_dir, f'{scan_id}.png')
    print(f'Generating structural connectome: {output_path}')
    plot_connectome(matrix=struct_mat, output_file=output_path, cmap='viridis')

# === Functional connectomes ===
modes = {
    'full': lambda m: m,
    'positive': lambda m: np.where(m > 0, m, 0),
    'negative': lambda m: np.where(m < 0, m, 0),
}

for scan_id in scan_ids:
    func_path = os.path.join(func_base_dir, map_func_filename(scan_id))
    try:
        func_mat = np.loadtxt(func_path, delimiter=',')
    except Exception as e:
        print(f'Error loading functional {func_path}: {e}')
        continue

    np.fill_diagonal(func_mat, 0)

    for mode, mask_fn in modes.items():
        masked_mat = mask_fn(func_mat)
        output_path = os.path.join(output_func_dir, f'{scan_id}_{mode}.png')
        print(f'Generating functional connectome ({mode}): {output_path}')
        plot_connectome(matrix=masked_mat, output_file=output_path, cmap='coolwarm')


# === Histograms for structural and functional connectomes ===
for scan_id in scan_ids:
    # Structural histogram
    try:
        struct_mat = np.loadtxt(os.path.join(struct_base_dir, map_struct_filename(scan_id)), delimiter=',')
    except Exception as e:
        print(f'Error loading structural matrix for histogram {scan_id}: {e}')
        continue
    np.fill_diagonal(struct_mat, 0)
    edges = upper_triangular_values(struct_mat)
    plot_histogram(
        data=edges,
        bins=50,
        title=f'Structural Connectome Edges Histogram ({scan_id})',
        xlabel='Edge Weight',
        output_file=os.path.join(output_hist_dir, f'struct_edges_{scan_id}.png')
    )

    # Functional histograms
    try:
        func_mat = np.loadtxt(os.path.join(func_base_dir, map_func_filename(scan_id)), delimiter=',')
    except Exception as e:
        print(f'Error loading functional matrix for histogram {scan_id}: {e}')
        continue
    np.fill_diagonal(func_mat, 0)
    for mode, mask_fn in modes.items():
        masked_mat = mask_fn(func_mat)
        edges = upper_triangular_values(masked_mat)
        if np.count_nonzero(edges) == 0:
            print(f'Skipping empty histogram for {scan_id} mode: {mode}')
            continue
        plot_histogram(
            data=edges,
            bins=50,
            title=f'Functional Connectome Edges Histogram ({scan_id} - {mode})',
            xlabel='Edge Weight',
            output_file=os.path.join(output_hist_dir, f'func_edges_{scan_id}_{mode}.png')
        )


# === Scatter plot: Structural vs Functional connectome edges ===
for scan_id in scan_ids:
    try:
        struct_mat = np.loadtxt(os.path.join(struct_base_dir, map_struct_filename(scan_id)), delimiter=',')
        func_mat = np.loadtxt(os.path.join(func_base_dir, map_func_filename(scan_id)), delimiter=',')
    except Exception as e:
        print(f'Error loading matrices for scatter {scan_id}: {e}')
        continue
    np.fill_diagonal(struct_mat, 0)
    np.fill_diagonal(func_mat, 0)
    struct_edges = upper_triangular_values(struct_mat)
    func_edges = upper_triangular_values(func_mat)
    output_path = os.path.join(output_scatter_dir, f'struct_vs_func_{scan_id}.png')
    print(f'Generating scatter plot structural vs functional: {output_path}')
    plot_scatter(
        x=struct_edges,
        y=func_edges,
        output_file=output_path,
        title=f'Structural vs Functional Connectome Edges ({scan_id})',
        xlabel='Structural Edge Weight',
        ylabel='Functional Edge Weight',
        fit_curve=True,
        curve_degree=1
    )


# === Scatter plot: Age vs Total Brain Volume by sex ===
demo_df = pd.read_csv(demo_file)
demo_df_first_scan = demo_df[demo_df['ID'].str.endswith('_1')]
male_df = demo_df_first_scan[demo_df_first_scan['sex'] == 'Male']
female_df = demo_df_first_scan[demo_df_first_scan['sex'] == 'Female']

output_path = os.path.join(output_scatter_dir, 'age_vs_brain_volume.png')
print(f'Generating scatter plot age vs brain volume: {output_path}')

plt.figure(figsize=(8,6))
plt.scatter(male_df['age'], male_df['total'], color='blue', label='Male', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.scatter(female_df['age'], female_df['total'], color='red', label='Female', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Age vs Total Brain Volume by Sex')
plt.xlabel('Age')
plt.ylabel('Total Brain Volume')
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.close()


# === Scatter plots: Scores related ===
scores_df = pd.read_csv(scores_file)
male_scores = scores_df[scores_df['Sex'] == 'Male']
female_scores = scores_df[scores_df['Sex'] == 'Female']

# Set your actual cognitive and processing speed column names here:
cog_score_col = 'Reasoning_raw'  # replace if needed
processing_speed_col = 'Processing_Speed_raw'               # replace if needed
age_col = 'Age'

# Age vs Cognitive Score
output_path = os.path.join(output_scatter_dir, 'age_vs_cognitive_score.png')
print(f'Generating scatter plot age vs cognitive score: {output_path}')

plt.figure(figsize=(8,6))
plt.scatter(male_scores[age_col], male_scores[cog_score_col], color='blue', label='Male', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.scatter(female_scores[age_col], female_scores[cog_score_col], color='red', label='Female', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Age vs Cognitive Score by Sex')
plt.xlabel('Age')
plt.ylabel('Cognitive Score')
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.close()

# Processing Speed vs Cognitive Score
output_path = os.path.join(output_scatter_dir, 'processing_speed_vs_cognitive_score.png')
print(f'Generating scatter plot processing speed vs cognitive score: {output_path}')

plt.figure(figsize=(8,6))
plt.scatter(male_scores[processing_speed_col], male_scores[cog_score_col], color='blue', label='Male', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.scatter(female_scores[processing_speed_col], female_scores[cog_score_col], color='red', label='Female', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Processing Speed vs Cognitive Score by Sex')
plt.xlabel('Processing Speed')
plt.ylabel('Cognitive Score')
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.close()

node_strength_data = []
scan_labels = []

for scan_id in scan_ids:
    try:
        mat = np.loadtxt(os.path.join(struct_base_dir, map_struct_filename(scan_id)), delimiter=',')
        np.fill_diagonal(mat, 0)
        node_strength = mat.sum(axis=1)
        node_strength_data.append(node_strength)
        scan_labels.append(scan_id)
    except Exception as e:
        print(f"Error loading {scan_id} for node strength: {e}")

output_path = os.path.join(output_hist_dir, 'node_strength_boxplot.png')
plot_boxplot(
    data_arrays=node_strength_data,
    labels=scan_labels,
    title='Node Strengths Across Scans',
    ylabel='Node Strength',
    output_file=output_path
)

male_ages = male_df['age'].dropna()
female_ages = female_df['age'].dropna()

output_path = os.path.join(output_hist_dir, 'age_boxplot_sex.png')
plot_boxplot(
    data_arrays=[male_ages, female_ages],
    labels=['Male', 'Female'],
    title='Age Distribution by Sex',
    ylabel='Age',
    output_file=output_path
)

volume_cols = ['corticalGM', 'subcortical', 'WM', 'total']

male_volumes = [male_df[col].dropna() for col in volume_cols]
female_volumes = [female_df[col].dropna() for col in volume_cols]

grouped_data = male_volumes + female_volumes
group_labels = [f'{col} (M)' for col in volume_cols] + [f'{col} (F)' for col in volume_cols]

output_path = os.path.join(output_hist_dir, 'brain_volume_boxplot_sex.png')
plot_boxplot(
    data_arrays=grouped_data,
    labels=group_labels,
    title='Brain Volumes by Sex',
    ylabel='Volume',
    output_file=output_path
)

# Replace if needed
fluid_col = 'Composite_average_Vocabulary_Naming'

male_fluid = male_scores[fluid_col].dropna()
female_fluid = female_scores[fluid_col].dropna()

output_path = os.path.join(output_hist_dir, 'fluid_intelligence_boxplot_sex.png')
plot_boxplot(
    data_arrays=[male_fluid, female_fluid],
    labels=['Male', 'Female'],
    title='Fluid Intelligence by Sex',
    ylabel='Score',
    output_file=output_path
)

output_path = os.path.join(output_hist_dir, 'node_strength_violin.png')
plot_violin(
    data_arrays=node_strength_data,
    labels=scan_labels,
    title='Node Strengths Across Scans (Violin)',
    ylabel='Node Strength',
    output_file=output_path
)

output_path = os.path.join(output_hist_dir, 'age_violin_sex.png')
plot_violin(
    data_arrays=[male_ages, female_ages],
    labels=['Male', 'Female'],
    title='Age Distribution by Sex (Violin)',
    ylabel='Age',
    output_file=output_path
)

output_path = os.path.join(output_hist_dir, 'brain_volume_violin_sex.png')
plot_violin(
    data_arrays=grouped_data,
    labels=group_labels,
    title='Brain Volumes by Sex (Violin)',
    ylabel='Volume',
    output_file=output_path
)

output_path = os.path.join(output_hist_dir, 'fluid_intelligence_violin_sex.png')
plot_violin(
    data_arrays=[male_fluid, female_fluid],
    labels=['Male', 'Female'],
    title='Fluid Intelligence by Sex (Violin)',
    ylabel='Score',
    output_file=output_path
)











