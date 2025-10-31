"""
Difference-in-Difference (DiD) Analysis
This script performs DiD analysis on the provided dataset to estimate
the causal impact of government subsidy on average wages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the data
print("Loading data...")
df = pd.read_csv('did_data.csv')

print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# Part a) Construct 'treated' variable
print("\n" + "="*80)
print("PART A: Creating 'treated' variable")
print("="*80)

# A region is treated if it ever receives treatment (treatment == 1)
df['treated'] = df.groupby('region_id')['treatment'].transform('max')

print("\nTreated variable created:")
print(f"Number of treated regions: {df[df['treated']==1]['region_id'].nunique()}")
print(f"Number of control regions: {df[df['treated']==0]['region_id'].nunique()}")
print("\nDistribution of treated variable:")
print(df['treated'].value_counts())

# Part b) Construct 'post' variable
print("\n" + "="*80)
print("PART B: Creating 'post' variable")
print("="*80)

# Post takes value 1 for 2010 or later, 0 for 2006-2009
df['post'] = (df['year'] >= 2010).astype(int)

print("\nPost variable created:")
print(f"Pre-treatment years (post=0): {df[df['post']==0]['year'].unique()}")
print(f"Post-treatment years (post=1): {df[df['post']==1]['year'].unique()}")
print("\nDistribution of post variable:")
print(df['post'].value_counts())

# Part c) Plot average wage over time for treated vs control groups
print("\n" + "="*80)
print("PART C: Plotting parallel trends")
print("="*80)

# Calculate average wage by year and treatment status
avg_wage_by_year = df.groupby(['year', 'treated'])['avg_wage'].mean().reset_index()

# Create the plot
plt.figure(figsize=(12, 7))
for treated_status in [0, 1]:
    data = avg_wage_by_year[avg_wage_by_year['treated'] == treated_status]
    label = 'Treated' if treated_status == 1 else 'Control'
    plt.plot(data['year'], data['avg_wage'], marker='o', linewidth=2, 
             markersize=8, label=label)

# Add vertical line at treatment time
plt.axvline(x=2009.5, color='red', linestyle='--', linewidth=2, 
            label='Treatment Start', alpha=0.7)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Wage', fontsize=12)
plt.title('Parallel Trends: Average Wage Over Time by Treatment Group', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('parallel_trends_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'parallel_trends_plot.png'")

# Analysis of parallel trends
print("\nParallel Trends Assessment (Pre-treatment period 2006-2009):")
pre_treatment = df[df['year'] < 2010]
pre_avg = pre_treatment.groupby(['year', 'treated'])['avg_wage'].mean().reset_index()
print(pre_avg.pivot(index='year', columns='treated', values='avg_wage'))

# Part d) Estimate DiD regression model
print("\n" + "="*80)
print("PART D: Difference-in-Difference Regression (Basic Model)")
print("="*80)

# Basic DiD model: avg_wage = β0 + β1*treated + β2*post + β3*(treated*post) + ε
# β3 is the DiD estimator (causal effect)

# Create interaction term
df['treated_post'] = df['treated'] * df['post']

# Estimate the model
model_basic = ols('avg_wage ~ treated + post + treated_post', data=df).fit()

print("\nBasic DiD Regression Results:")
print(model_basic.summary())

print("\n" + "-"*80)
print("INTERPRETATION:")
print("-"*80)
did_estimate = model_basic.params['treated_post']
print(f"\nDiD Estimate (β3): {did_estimate:.4f}")
print(f"Standard Error: {model_basic.bse['treated_post']:.4f}")
print(f"P-value: {model_basic.pvalues['treated_post']:.4f}")

if model_basic.pvalues['treated_post'] < 0.05:
    print(f"\nThe DiD estimate is statistically significant at the 5% level.")
    if did_estimate > 0:
        print(f"The subsidy INCREASES average wages by approximately ${did_estimate:.2f}.")
    else:
        print(f"The subsidy DECREASES average wages by approximately ${abs(did_estimate):.2f}.")
else:
    print(f"\nThe DiD estimate is NOT statistically significant at the 5% level.")
    print("We cannot conclude that the subsidy has a causal effect on wages.")

# Part e) DiD with control variables
print("\n" + "="*80)
print("PART E: Difference-in-Difference with Control Variables")
print("="*80)

# DiD model with controls
formula = ('avg_wage ~ treated + post + treated_post + '
           'population + unemployment_rate + gdp_per_capita + '
           'exports_per_capita + fdi_inflow')

model_controls = ols(formula, data=df).fit()

print("\nDiD Regression with Control Variables:")
print(model_controls.summary())

print("\n" + "-"*80)
print("INTERPRETATION WITH CONTROLS:")
print("-"*80)
did_estimate_controls = model_controls.params['treated_post']
print(f"\nDiD Estimate with controls (β3): {did_estimate_controls:.4f}")
print(f"Standard Error: {model_controls.bse['treated_post']:.4f}")
print(f"P-value: {model_controls.pvalues['treated_post']:.4f}")

print(f"\nComparison:")
print(f"  Basic DiD estimate: {did_estimate:.4f}")
print(f"  DiD estimate with controls: {did_estimate_controls:.4f}")
print(f"  Difference: {abs(did_estimate - did_estimate_controls):.4f}")

if model_controls.pvalues['treated_post'] < 0.05:
    print(f"\nThe DiD estimate with controls is statistically significant.")
    if did_estimate_controls > 0:
        print(f"After controlling for confounders, the subsidy INCREASES wages by ${did_estimate_controls:.2f}.")
    else:
        print(f"After controlling for confounders, the subsidy DECREASES wages by ${abs(did_estimate_controls):.2f}.")
else:
    print(f"\nThe DiD estimate with controls is NOT statistically significant.")

# Part f) Heterogeneous treatment effects by sector
print("\n" + "="*80)
print("PART F: Heterogeneous Treatment Effects by Sector")
print("="*80)

sectors = df['sector'].unique()
results_by_sector = {}

for sector in sectors:
    print(f"\n{'='*80}")
    print(f"Sector: {sector}")
    print(f"{'='*80}")
    
    # Subset data for this sector
    sector_data = df[df['sector'] == sector].copy()
    
    # Estimate DiD for this sector
    model_sector = ols(formula, data=sector_data).fit()
    
    # Store results
    did_coef = model_sector.params['treated_post']
    did_se = model_sector.bse['treated_post']
    did_pval = model_sector.pvalues['treated_post']
    
    results_by_sector[sector] = {
        'coefficient': did_coef,
        'std_error': did_se,
        'p_value': did_pval,
        'significant': did_pval < 0.05
    }
    
    print(f"\nDiD Estimate: {did_coef:.4f}")
    print(f"Standard Error: {did_se:.4f}")
    print(f"P-value: {did_pval:.4f}")
    print(f"Significant at 5% level: {did_pval < 0.05}")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: Heterogeneous Treatment Effects Across Sectors")
print("="*80)

summary_df = pd.DataFrame(results_by_sector).T
summary_df.columns = ['Coefficient', 'Std Error', 'P-value', 'Significant']
print("\n", summary_df)

# Visualize heterogeneous effects
plt.figure(figsize=(12, 7))
sectors_list = list(results_by_sector.keys())
coefficients = [results_by_sector[s]['coefficient'] for s in sectors_list]
std_errors = [results_by_sector[s]['std_error'] for s in sectors_list]

# Create bar plot with error bars
x_pos = np.arange(len(sectors_list))
colors = ['green' if results_by_sector[s]['significant'] else 'gray' for s in sectors_list]

plt.bar(x_pos, coefficients, yerr=std_errors, capsize=10, 
        alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

plt.xlabel('Sector', fontsize=12)
plt.ylabel('DiD Estimate (Treatment Effect)', fontsize=12)
plt.title('Heterogeneous Treatment Effects by Sector\n(Green = Significant at 5% level, Gray = Not Significant)', 
          fontsize=14, fontweight='bold')
plt.xticks(x_pos, sectors_list)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('heterogeneous_effects_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'heterogeneous_effects_plot.png'")

print("\n" + "="*80)
print("VARIATION IN TREATMENT EFFECTS:")
print("="*80)
coef_values = [v['coefficient'] for v in results_by_sector.values()]
print(f"Mean effect across sectors: {np.mean(coef_values):.4f}")
print(f"Std deviation of effects: {np.std(coef_values):.4f}")
print(f"Range: [{min(coef_values):.4f}, {max(coef_values):.4f}]")
print(f"Difference between max and min: {max(coef_values) - min(coef_values):.4f}")

# Find most and least affected sectors
max_sector = max(results_by_sector, key=lambda x: results_by_sector[x]['coefficient'])
min_sector = min(results_by_sector, key=lambda x: results_by_sector[x]['coefficient'])

print(f"\nMost positively affected sector: {max_sector} ({results_by_sector[max_sector]['coefficient']:.4f})")
print(f"Least affected sector: {min_sector} ({results_by_sector[min_sector]['coefficient']:.4f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  - parallel_trends_plot.png")
print("  - heterogeneous_effects_plot.png")
