import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Results
df = pd.read_csv("ADEMP_Factorial_Results.csv")

# Aggregate
agg_df = df.groupby(['Design', 'Planted_Gamma'])[['T3_Power', 'T4_Power', 'T6_Power']].mean().reset_index()

# Plot Setup
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot P (Paired)
sns.lineplot(data=agg_df[agg_df['Design'] == 'P'], x='Planted_Gamma', y='T4_Power', ax=axes[0], marker="o", label="T4 (Nested LRT)", linewidth=2.5)
sns.lineplot(data=agg_df[agg_df['Design'] == 'P'], x='Planted_Gamma', y='T3_Power', ax=axes[0], marker="s", label="T3 (Conditional Perm)", linewidth=2.5)
sns.lineplot(data=agg_df[agg_df['Design'] == 'P'], x='Planted_Gamma', y='T6_Power', ax=axes[0], marker="^", label="T6 (GCM)", linewidth=2.5)
axes[0].axhline(0.05, ls='--', color='red', label="Alpha (0.05)")
axes[0].set_title("True Paired Data (P)", fontsize=14, fontweight='bold')
axes[0].set_xlabel(r"Planted Biological Signal ($\gamma$)", fontsize=12)
axes[0].set_ylabel("Power / Rejection Rate", fontsize=12)
axes[0].set_ylim([-0.05, 1.05])

# Plot L0 (PSM-Linked)
sns.lineplot(data=agg_df[agg_df['Design'] == 'L0'], x='Planted_Gamma', y='T4_Power', ax=axes[1], marker="o", label="T4 (Nested LRT)", linewidth=2.5)
sns.lineplot(data=agg_df[agg_df['Design'] == 'L0'], x='Planted_Gamma', y='T3_Power', ax=axes[1], marker="s", label="T3 (Conditional Perm)", linewidth=2.5)
sns.lineplot(data=agg_df[agg_df['Design'] == 'L0'], x='Planted_Gamma', y='T6_Power', ax=axes[1], marker="^", label="T6 (GCM)", linewidth=2.5)
axes[1].axhline(0.05, ls='--', color='red', label="Alpha (0.05)")
axes[1].set_title("Cross-Cohort PSM Linked Data (L0)", fontsize=14, fontweight='bold')
axes[1].set_xlabel(r"Planted Biological Signal ($\gamma$)", fontsize=12)

# Adjust legends
axes[0].legend(loc="upper left")
axes[1].legend().remove()

plt.tight_layout()
plt.savefig("figure1_rejection_rates.png", dpi=300)
print("Saved figure1_rejection_rates.png")

