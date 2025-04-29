import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading ---
try:
    df = pd.read_csv("es_results_bert-base-spanish-wwm-cased-filtered.csv")
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# --- Data Cleaning ---
df = df.rename(columns={
    'suggestion_last_translation_result': 'KendallTau_RDM',
    'HTER': 'HTER'
})

# Clean the KendallTau_RDM column: remove brackets and convert to numeric
try:
    # Remove brackets using string accessor (.str)
    df['KendallTau_RDM'] = df['KendallTau_RDM'].astype(str).str.strip('[]')
    # Convert to numeric (float)
    df['KendallTau_RDM'] = pd.to_numeric(df['KendallTau_RDM'])
except KeyError:
    print("Error: Column 'suggestion_last_translation_result' not found. Check column names.")
    print(f"Columns available: {df.columns.tolist()}")
    exit()
except Exception as e:
    print(f"Error cleaning 'KendallTau_RDM' column: {e}")
    exit()

# Ensure HTER is numeric
try:
    df['HTER'] = pd.to_numeric(df['HTER'])
except KeyError:
    print("Error: Column 'HTER' not found.")
    exit()
except Exception as e:
    print(f"Error converting 'HTER' column to numeric: {e}")
    exit()

# Display cleaned data info
print("Cleaned Data Head:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("-" * 30)

# Drop rows with NaN values that might result from cleaning errors
df_cleaned = df[['KendallTau_RDM', 'HTER']].dropna()

if df_cleaned.empty:
    print("Error: No valid data remaining after cleaning and dropping NaNs.")
    exit()

print(f"Using {len(df_cleaned)} rows for analysis after cleaning.")
print("-" * 30)




# --- Correlation Analysis ---
# Calculate Pearson's correlation coefficient and p-value
pearson_corr, pearson_p_value = stats.pearsonr(df_cleaned['KendallTau_RDM'], df_cleaned['HTER'])
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"Pearson P-value: {pearson_p_value:.4f}")

# Calculate Spearman's rank correlation coefficient and p-value
spearman_corr, spearman_p_value = stats.spearmanr(df_cleaned['KendallTau_RDM'], df_cleaned['HTER'])
print(f"\nSpearman Rank Correlation: {spearman_corr:.4f}")
print(f"Spearman P-value: {spearman_p_value:.4f}")
print("-" * 30)

# --- Interpretation ---
alpha = 0.05 # Significance level
print(f"Significance level (alpha): {alpha}")

print("\nInterpretation (Pearson):")
if pearson_p_value < alpha:
    print(f"The correlation ({pearson_corr:.4f}) is statistically significant (p < {alpha}).")
    if pearson_corr > 0:
        print("There is a significant positive linear relationship.")
    elif pearson_corr < 0:
        print("There is a significant negative linear relationship (as expected).")
    else:
        print("Correlation is near zero.")
else:
    print(f"The correlation ({pearson_corr:.4f}) is NOT statistically significant (p >= {alpha}).")
    print("We cannot conclude there is a linear relationship based on this test.")

print("\nInterpretation (Spearman):")
if spearman_p_value < alpha:
     print(f"The rank correlation ({spearman_corr:.4f}) is statistically significant (p < {alpha}).")
     if spearman_corr > 0:
        print("There is a significant positive monotonic relationship.")
     elif spearman_corr < 0:
        print("There is a significant negative monotonic relationship (as expected).")
     else:
        print("Rank correlation is near zero.")
else:
    print(f"The rank correlation ({spearman_corr:.4f}) is NOT statistically significant (p >= {alpha}).")
    print("We cannot conclude there is a monotonic relationship based on this test.")
print("-" * 30)

# --- Plotting ---
plt.figure(figsize=(10, 6))
sns.regplot(data=df_cleaned, x='KendallTau_RDM', y='HTER',
            scatter_kws={'alpha':0.6, 's':50},
            line_kws={'color':'red', 'linewidth':2})

plt.title('Relationship between RDM Similarity (Kendall-Tau) and HTER')
plt.xlabel('RDM Similarity (Kendall-Tau Score)')
plt.ylabel('Post-Editing Effort (HTER Score)')
plt.grid(True, linestyle='--', alpha=0.6)

# --- Correctly using the variables in an f-string ---
# This takes the VALUE stored in the variable spearman_corr
# and formats it to 3 decimal places (.3f), and similarly for the p-value.
plt.text(0.05, 0.95, f"Spearman ρ: {spearman_corr:.3f}\np-value: {spearman_p_value:.3g}",
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
# --- End of correct usage ---

plt.tight_layout()
plt.show()