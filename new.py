import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, wilcoxon
from statannotations.Annotator import Annotator

# ---------- CONFIG ----------
demographics_file = "demographics.csv"
connectome_dir = Path("connectomes")  # folder with all connectomes
num_nodes = 86                        # number of ROIs in each connectome
normalize = True                       # normalize each connectome to max edge = 1
output_ns_csv = "ns_group_comparisons_s1.csv"
output_nns_csv = "nns_scores.csv"
# ----------------------------

demo = pd.read_csv("demographics.csv")

# Keep only session 1
demo_s1 = demo[demo["Subject"].str.endswith("_s1")].copy()

# Split group
controls = demo_s1[demo_s1["Group"] == "CON"]
patients = demo_s1[demo_s1["Group"] == "S1"]

def summarize_group(df, group_name):
    return {
        "Group": group_name,
        "N": len(df),
        "Age (mean±SD)": f"{df['Age'].mean():.1f} ± {df['Age'].std():.1f}" if "Age" in df else "NA",
        "Sex (M/F)": f"{(df['Gender']=='M').sum()}/{(df['Gender']=='F').sum()}" if "Gender" in df else "NA",
         "PTA (mean±SD)": (
            f"{pd.to_numeric(df['PTA Duration'], errors='coerce').mean():.1f} ± {pd.to_numeric(df['PTA Duration'], errors='coerce').std():.1f}"
            if group_name == "TBI" else "NA")
    }

table = pd.DataFrame([
    summarize_group(controls, "Control"),
    summarize_group(patients, "TBI")
])

print(table)

# --- Cache for loaded connectomes ---
connectome_cache = {}

def load_connectome(subject_id):
    """Load subject connectome as numpy array (with caching)."""
    if subject_id in connectome_cache:
        return connectome_cache[subject_id]

    file_path = connectome_dir / f"{subject_id}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Connectome not found for {subject_id}")
    
    mat = np.loadtxt(file_path)
    if normalize:
        m = np.max(np.abs(mat))
        if m > 0:
            mat = mat / m
    
    connectome_cache[subject_id] = mat
    return mat

# --- Network similarity using edge-wise correlation ---
def network_similarity(A, B):
    """Compute network similarity as correlation of edge weights."""
    A = A.copy()
    B = B.copy()
    np.fill_diagonal(A, 0)
    np.fill_diagonal(B, 0)
    return np.corrcoef(A.flatten(), B.flatten())[0,1]

# --- Load demographics ---
demo = pd.read_csv(demographics_file)
demo['Subject'] = demo['Subject'].str.strip()

# --- Identify groups ---
healthy_s1 = demo[(demo['Group'] == 'CON') & (demo['Subject'].str.endswith('_s1'))]['Subject'].tolist()
tbi_s1 = demo[(demo['Group'] != 'CON') & (demo['Subject'].str.endswith('_s1'))]['Subject'].tolist()

print("Healthy S1 subjects:", healthy_s1)
print("TBI S1 subjects:", tbi_s1)

# --- Function to compute pairwise NS between two lists ---
def compare_groups(list1, list2, label):
    results = []
    for sub1 in list1:
        try:
            A = load_connectome(sub1)
        except Exception as e:
            print(f"Skipping {sub1}: {e}")
            continue

        for sub2 in list2:
            # Avoid comparing the same subject with itself in same-group comparisons
            if label != "CON–TBI" and sub1 == sub2:
                continue
            try:
                B = load_connectome(sub2)
            except Exception as e:
                print(f"Skipping {sub2}: {e}")
                continue

            ns = network_similarity(A, B)
            results.append({
                "Subject1": sub1,
                "Subject2": sub2,
                "Group_Comparison": label,
                "NS": ns
            })
    return results

table = pd.DataFrame({
    "Group": ["Control", "TBI"],
    "N": [35, 34],
    "Age (mean±SD)": ["35.0 ± 10.3", "35.0 ± 15.2"],
    "Sex (M/F)": ["26/9", "22/12"],
    "PTA (mean±SD)": ["NA", "25.2 ± 21.3"]
})

# Create matplotlib table
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis("off")
tbl = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="center", loc="center")

# Style the table
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

# Save to file
output_path = "demographics_table_final.png"
plt.savefig(output_path, bbox_inches="tight", dpi=300)
output_path

# --- Compute all three comparisons ---
results_all = []
results_all.extend(compare_groups(healthy_s1, healthy_s1, "CON–CON"))
results_all.extend(compare_groups(healthy_s1, tbi_s1, "CON–TBI"))
results_all.extend(compare_groups(tbi_s1, tbi_s1, "TBI–TBI"))

df_results = pd.DataFrame(results_all)
df_results.to_csv(output_ns_csv, index=False)
print(f"Saved pairwise NS results to {output_ns_csv}")

# --- NS violin plot ---
if not df_results.empty:
    plt.figure(figsize=(8,5))
    import seaborn as sns
    sns.violinplot(x='Group_Comparison', y='NS', data=df_results)
    plt.ylabel("Network Similarity (NS)")
    plt.title("Pairwise Network Similarity Across Groups (S1)")
    plt.ylim(0.6,1)
    plt.show()
else:
    print("No pairwise NS computed — check that connectome files match Subject IDs")

# --- Compute NNS scores ---
# NNSH: healthy relative to healthy
nsh_df = df_results[df_results['Group_Comparison'] == 'CON–CON']
nnsH = nsh_df.groupby('Subject1')['NS'].mean().reset_index().rename(columns={'NS':'NNSH'})

# NNSPH: patient relative to healthy
nsph_df = df_results[df_results['Group_Comparison'] == 'CON–TBI']
nnsPH = nsph_df.groupby('Subject2')['NS'].mean().reset_index().rename(columns={'NS':'NNSPH'})

# NNSP: patient relative to patients
nsp_df = df_results[df_results['Group_Comparison'] == 'TBI–TBI']
nnsP = nsp_df.groupby('Subject1')['NS'].mean().reset_index().rename(columns={'NS':'NNSP'})

# Merge into one dataframe
nns_df = pd.merge(nnsH, nnsPH, left_on='Subject1', right_on='Subject2', how='outer')
nns_df = pd.merge(nns_df, nnsP, left_on='Subject1', right_on='Subject1', how='outer')
nns_df.drop(columns=['Subject2'], inplace=True)
nns_df = nns_df.rename(columns={'Subject1':'Subject'})

# Save NNS
nns_df.to_csv(output_nns_csv, index=False)
print(f"Saved NNS scores to {output_nns_csv}")
print(nns_df.head())

# --- NNS violin plots ---
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.violinplot(nns_df['NNSH'].dropna())
plt.title("NNSH (Healthy vs Healthy)")
plt.ylabel("Network Normality Score")
plt.ylim(0.7,1)

plt.subplot(1,3,2)
plt.violinplot(nns_df['NNSPH'].dropna())
plt.title("NNSPH (Patient vs Healthy)")
plt.ylim(0.7,1)

plt.subplot(1,3,3)
plt.violinplot(nns_df['NNSP'].dropna())
plt.title("NNSP (Patient vs Patient)")
plt.ylim(0.7,1)

plt.tight_layout()
plt.show()

con_con = df_results[df_results["Group_Comparison"]=="CON–CON"]["NS"]
con_tbi = df_results[df_results["Group_Comparison"]=="CON–TBI"]["NS"]
tbi_tbi = df_results[df_results["Group_Comparison"]=="TBI–TBI"]["NS"]

# Mann–Whitney U tests
stat1, p1 = mannwhitneyu(con_con, con_tbi, alternative="two-sided")
stat2, p2 = mannwhitneyu(con_con, tbi_tbi, alternative="two-sided")
stat3, p3 = mannwhitneyu(con_tbi, tbi_tbi, alternative="two-sided")

print("CON–CON vs CON–TBI: p =", p1)
print("CON–CON vs TBI–TBI: p =", p2)
print("CON–TBI vs TBI–TBI: p =", p3)

nns_long = pd.DataFrame({
    "Group": (["NNSH"] * len(nns_df['NNSH'].dropna())
             + ["NNSPH"] * len(nns_df['NNSPH'].dropna())
             + ["NNSP"] * len(nns_df['NNSP'].dropna())),
    "NNS": (list(nns_df['NNSH'].dropna())
           + list(nns_df['NNSPH'].dropna())
           + list(nns_df['NNSP'].dropna()))
})

# Comparisons
comparisons = [("NNSH", "NNSPH"), ("NNSH", "NNSP"), ("NNSPH", "NNSP")]
effect_sizes = {}

def compute_es(x, y, test="MW"):
    """Compute effect size ES = Z/sqrt(N)."""
    n1, n2 = len(x), len(y)
    if test == "MW":
        u, _ = mannwhitneyu(x, y, alternative="two-sided")
        mean_u = n1*n2/2
        std_u = np.sqrt(n1*n2*(n1+n2+1)/12)
        z = (u - mean_u)/std_u
        es = z/np.sqrt(n1+n2)
    elif test == "W":
        stat, _ = wilcoxon(x, y)
        # Normal approx for Wilcoxon
        mean_w = n1*(n1+1)/4
        std_w = np.sqrt(n1*(n1+1)*(2*n1+1)/24)
        z = (stat - mean_w)/std_w
        es = z/np.sqrt(n1)
    return es

# Compute ES for each pair
for g1, g2 in comparisons:
    x = nns_long[nns_long["Group"]==g1]["NNS"].values
    y = nns_long[nns_long["Group"]==g2]["NNS"].values
    test = "W" if g1=="NNSPH" and g2=="NNSP" else "MW"
    es = compute_es(x, y, test=test)
    if (es < 0):
        es = es * -1
    effect_sizes[(g1,g2)] = es
    print(f"{g1} vs {g2}: ES = {es:.3f}")

# Plot violins + swarm
plt.figure(figsize=(8,5))
ax = sns.violinplot(x="Group", y="NNS", data=nns_long, inner=None, cut=0)
sns.swarmplot(x="Group", y="NNS", data=nns_long, color="black", alpha=0.5, size=3)

# Annotate ES above comparisons
y_max = nns_long["NNS"].max()
y_step = 0.02
for i, ((g1, g2), es) in enumerate(effect_sizes.items()):
    x1, x2 = list(nns_long["Group"].unique()).index(g1), list(nns_long["Group"].unique()).index(g2)
    ax.plot([x1, x1, x2, x2], [y_max+i*y_step, y_max+(i+0.5)*y_step, y_max+(i+0.5)*y_step, y_max+i*y_step], lw=1.2, c='k')
    ax.text((x1+x2)/2, y_max+(i+0.6)*y_step, f"ES={es:.2f}", ha='center', va='bottom', color='k')

plt.ylim(0.7, 1.05)
plt.ylabel("Network Normality Score")
plt.title("Network Normality Score Across Groups (Effect Sizes Shown)")
plt.tight_layout()
plt.show()


df_long = pd.DataFrame({
    'Subject': pd.concat([df_results['Subject1'], df_results['Subject2']]),
    'Group_Comparison': pd.concat([df_results['Group_Comparison'], df_results['Group_Comparison']]),
    'NS': pd.concat([df_results['NS'], df_results['NS']])
})

# Mixed effects model: random intercept for each Subject
md = sm.MixedLM.from_formula("NS ~ C(Group_Comparison)", groups="Subject", data=df_long)
mdf = md.fit()
print(mdf.summary())
# -----------------------
# 2. Fixed Effects Model
# -----------------------
# Includes dummy variables for each subject to control for subject-specific effects
model_fe = smf.ols(
    "NS ~ C(Group_Comparison) + C(Subject1)",
    data=df_results
).fit()

print("\n=== Fixed Effects Model ===")
print(model_fe.summary())

# Save fixed effects results to CSV
results_df = pd.DataFrame({
    "Term": model_fe.params.index,
    "Estimate": model_fe.params.values,
    "Std. Error": model_fe.bse.values,
    "t-value": model_fe.tvalues.values,
    "p-value": model_fe.pvalues.values
})
results_df.to_csv("fixed_effects_results.csv", index=False)
print("\nFixed effects results saved to fixed_effects_results.csv")

fe_params = mdf.fe_params
fe_cov = mdf.cov_params()  # covariance of estimates

# Compute means and standard errors
groups = ['CON–CON', 'CON–TBI', 'TBI–TBI']
means = [
    fe_params['Intercept'],
    fe_params['Intercept'] + fe_params.get('C(Group_Comparison)[T.CON–TBI]', 0),
    fe_params['Intercept'] + fe_params.get('C(Group_Comparison)[T.TBI–TBI]', 0)
]

# Compute approximate SE for each group
# SE(CON–CON) = SE(intercept)
# SE(CON–TBI) = sqrt(Var(intercept) + Var(C(Group_Comparison)[T.CON–TBI]) + 2*Cov)
ses = [
    np.sqrt(fe_cov.loc['Intercept','Intercept']),
    np.sqrt(
        fe_cov.loc['Intercept','Intercept'] +
        fe_cov.loc['C(Group_Comparison)[T.CON–TBI]','C(Group_Comparison)[T.CON–TBI]'] +
        2*fe_cov.loc['Intercept','C(Group_Comparison)[T.CON–TBI]']
    ),
    np.sqrt(
        fe_cov.loc['Intercept','Intercept'] +
        fe_cov.loc['C(Group_Comparison)[T.TBI–TBI]','C(Group_Comparison)[T.TBI–TBI]'] +
        2*fe_cov.loc['Intercept','C(Group_Comparison)[T.TBI–TBI]']
    )
]

# Plot
plt.figure(figsize=(6,5))
plt.bar(groups, means, yerr=[1.96*np.array(ses)], capsize=5, color=['skyblue','salmon','lightgreen'])
plt.ylabel("Estimated Network Similarity (NS)")
plt.title("Estimated NS by Group Comparison")
plt.ylim(0.8, 0.86)
plt.show()
