"""
Regression Discontinuity Design (RDD) Analysis
This script performs RDD analysis to estimate the causal effect of
scholarship on 10th standard test scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the data
print("Loading data...")
df = pd.read_csv('rdd_data.csv')

print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Part a) Create treatment variable D based on cutoff rule
print("\n" + "="*80)
print("PART A: Creating Treatment Variable D")
print("="*80)

# Students who score more than 0 get scholarship
df['D'] = (df['5th_score'] > 0).astype(int)

print("\nTreatment variable D created (D=1 if 5th_score > 0, else D=0):")
print(f"Number of students who received scholarship (D=1): {df['D'].sum()}")
print(f"Number of students who did not receive scholarship (D=0): {(1-df['D']).sum()}")
print(f"Total students: {len(df)}")
print(f"\nProportion receiving scholarship: {df['D'].mean():.2%}")

# Verify cutoff
print("\nVerification around cutoff:")
print(f"Min score among treated (D=1): {df[df['D']==1]['5th_score'].min():.4f}")
print(f"Max score among untreated (D=0): {df[df['D']==0]['5th_score'].max():.4f}")

# Part b) Plot continuity of hours_studied and mother_edu around cutoff
print("\n" + "="*80)
print("PART B: Testing Continuity of Covariates at Cutoff")
print("="*80)

# Create bins for visualization
bins = np.linspace(df['5th_score'].min(), df['5th_score'].max(), 30)
df['score_bin'] = pd.cut(df['5th_score'], bins=bins)
df['score_bin_center'] = df['score_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

# Plot 1: hours_studied vs 5th_score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Hours studied
bin_hours = df.groupby('score_bin_center')['hours_studied'].mean()
ax1.scatter(df['5th_score'], df['hours_studied'], alpha=0.3, s=20, label='Individual observations')
ax1.plot(bin_hours.index, bin_hours.values, 'r-', linewidth=3, label='Bin averages')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Cutoff (score=0)')
ax1.set_xlabel('5th Grade Score', fontsize=12)
ax1.set_ylabel('Hours Studied', fontsize=12)
ax1.set_title('Continuity Test: Hours Studied vs 5th Score\n(No jump suggests continuity)', 
              fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mother's education
bin_mother = df.groupby('score_bin_center')['mother_edu'].mean()
ax2.scatter(df['5th_score'], df['mother_edu'], alpha=0.3, s=20, label='Individual observations')
ax2.plot(bin_mother.index, bin_mother.values, 'r-', linewidth=3, label='Bin averages')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Cutoff (score=0)')
ax2.set_xlabel('5th Grade Score', fontsize=12)
ax2.set_ylabel("Mother's Education (years)", fontsize=12)
ax2.set_title("Continuity Test: Mother's Education vs 5th Score\n(No jump suggests continuity)", 
              fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('continuity_test_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'continuity_test_plot.png'")

# Statistical tests for continuity
print("\nFormal continuity tests at cutoff:")

# Test 1: hours_studied
just_below = df[(df['5th_score'] > -0.5) & (df['5th_score'] <= 0)]
just_above = df[(df['5th_score'] > 0) & (df['5th_score'] <= 0.5)]

t_stat_hours, p_val_hours = stats.ttest_ind(just_below['hours_studied'], 
                                              just_above['hours_studied'])
print(f"\nHours Studied:")
print(f"  Mean just below cutoff: {just_below['hours_studied'].mean():.2f}")
print(f"  Mean just above cutoff: {just_above['hours_studied'].mean():.2f}")
print(f"  T-statistic: {t_stat_hours:.4f}")
print(f"  P-value: {p_val_hours:.4f}")
print(f"  Significant difference? {p_val_hours < 0.05}")

# Test 2: mother_edu
t_stat_mother, p_val_mother = stats.ttest_ind(just_below['mother_edu'], 
                                               just_above['mother_edu'])
print(f"\nMother's Education:")
print(f"  Mean just below cutoff: {just_below['mother_edu'].mean():.2f}")
print(f"  Mean just above cutoff: {just_above['mother_edu'].mean():.2f}")
print(f"  T-statistic: {t_stat_mother:.4f}")
print(f"  P-value: {p_val_mother:.4f}")
print(f"  Significant difference? {p_val_mother < 0.05}")

if p_val_hours > 0.05 and p_val_mother > 0.05:
    print("\n✓ Both covariates show no significant discontinuity at cutoff.")
    print("  This supports the validity of the RDD design (no manipulation).")
else:
    print("\n⚠ Warning: Some covariates show discontinuity, which may suggest manipulation.")

# Part c) Plot discontinuity in 10th_score vs 5th_score
print("\n" + "="*80)
print("PART C: Visualizing Discontinuity in Outcome")
print("="*80)

# Plot 10th score vs 5th score
plt.figure(figsize=(14, 8))

# Scatter plot with color by treatment
colors = ['red' if d == 0 else 'blue' for d in df['D']]
labels_done = {'red': False, 'blue': False}

for i, (x, y, c) in enumerate(zip(df['5th_score'], df['10th_score'], colors)):
    label = None
    if c == 'red' and not labels_done['red']:
        label = 'No Scholarship (D=0)'
        labels_done['red'] = True
    elif c == 'blue' and not labels_done['blue']:
        label = 'Scholarship (D=1)'
        labels_done['blue'] = True
    plt.scatter(x, y, alpha=0.4, s=30, c=c, label=label)

# Bin averages
bin_10th = df.groupby('score_bin_center')['10th_score'].mean()
plt.plot(bin_10th.index, bin_10th.values, 'black', linewidth=3, 
         label='Bin averages', zorder=5)

# Separate regression lines for left and right of cutoff
left_data = df[df['5th_score'] <= 0]
right_data = df[df['5th_score'] > 0]

if len(left_data) > 0:
    z_left = np.polyfit(left_data['5th_score'], left_data['10th_score'], 1)
    p_left = np.poly1d(z_left)
    x_left = np.linspace(left_data['5th_score'].min(), 0, 100)
    plt.plot(x_left, p_left(x_left), 'r--', linewidth=2, label='Linear fit (left)', alpha=0.8)

if len(right_data) > 0:
    z_right = np.polyfit(right_data['5th_score'], right_data['10th_score'], 1)
    p_right = np.poly1d(z_right)
    x_right = np.linspace(0, right_data['5th_score'].max(), 100)
    plt.plot(x_right, p_right(x_right), 'b--', linewidth=2, label='Linear fit (right)', alpha=0.8)

# Cutoff line
plt.axvline(x=0, color='green', linestyle='--', linewidth=3, label='Cutoff (score=0)')

plt.xlabel('5th Grade Normalized Score', fontsize=13)
plt.ylabel('10th Grade Test Score', fontsize=13)
plt.title('Regression Discontinuity: 10th Grade Score vs 5th Grade Score\n(Jump at cutoff shows treatment effect)', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rdd_discontinuity_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'rdd_discontinuity_plot.png'")

# Estimate visual discontinuity
if len(left_data) > 0 and len(right_data) > 0:
    jump = p_right(0) - p_left(0)
    print(f"\nVisual estimate of discontinuity (jump at cutoff): {jump:.2f} points")

# Part d) Estimate RDD model with covariates
print("\n" + "="*80)
print("PART D: Regression Discontinuity Estimation")
print("="*80)

# Create running variable centered at cutoff
df['running_var'] = df['5th_score']
df['running_var_centered'] = df['5th_score'] - 0  # Centered at cutoff

# Model 1: Basic RDD (sharp design)
print("\nModel 1: Basic RDD (without covariates)")
print("-" * 80)

# Simple model: Y = α + τ*D + β*X + ε
# where D is treatment, X is running variable
model_basic = ols('Q("10th_score") ~ D + running_var_centered', data=df).fit()
print(model_basic.summary())

tau_basic = model_basic.params['D']
print(f"\nTreatment Effect (τ) - Basic Model: {tau_basic:.4f}")
print(f"Standard Error: {model_basic.bse['D']:.4f}")
print(f"P-value: {model_basic.pvalues['D']:.4f}")

# Model 2: RDD with linear interaction (allowing different slopes)
print("\n" + "="*80)
print("Model 2: RDD with Different Slopes on Each Side")
print("-" * 80)

# Create interaction term
df['D_running'] = df['D'] * df['running_var_centered']

model_interaction = ols('Q("10th_score") ~ D + running_var_centered + D_running', 
                        data=df).fit()
print(model_interaction.summary())

tau_interaction = model_interaction.params['D']
print(f"\nTreatment Effect (τ) - With Interaction: {tau_interaction:.4f}")
print(f"Standard Error: {model_interaction.bse['D']:.4f}")
print(f"P-value: {model_interaction.pvalues['D']:.4f}")

# Model 3: RDD with all covariates (full model)
print("\n" + "="*80)
print("Model 3: RDD with All Covariates (Full Model)")
print("-" * 80)

# Full model with covariates
formula_full = ('Q("10th_score") ~ D + running_var_centered + D_running + '
                'hours_studied + mother_edu + female')

model_full = ols(formula_full, data=df).fit()
print(model_full.summary())

tau_full = model_full.params['D']
print(f"\nTreatment Effect (τ) - Full Model with Covariates: {tau_full:.4f}")
print(f"Standard Error: {model_full.bse['D']:.4f}")
print(f"P-value: {model_full.pvalues['D']:.4f}")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: Comparison of RDD Estimates")
print("="*80)

comparison_data = {
    'Model': ['Basic RDD', 'RDD with Interaction', 'RDD with Covariates'],
    'Treatment Effect': [tau_basic, tau_interaction, tau_full],
    'Std Error': [model_basic.bse['D'], model_interaction.bse['D'], model_full.bse['D']],
    'P-value': [model_basic.pvalues['D'], model_interaction.pvalues['D'], model_full.pvalues['D']],
    'R-squared': [model_basic.rsquared, model_interaction.rsquared, model_full.rsquared]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)

if model_full.pvalues['D'] < 0.05:
    print(f"\n✓ The scholarship has a STATISTICALLY SIGNIFICANT causal effect on 10th grade scores.")
    print(f"  Receiving the scholarship increases 10th grade test scores by {tau_full:.2f} points.")
    print(f"  This effect is significant at the 5% level (p-value = {model_full.pvalues['D']:.4f}).")
else:
    print(f"\n✗ The scholarship does NOT have a statistically significant effect.")
    print(f"  Point estimate: {tau_full:.2f} points")
    print(f"  P-value: {model_full.pvalues['D']:.4f} (not significant at 5% level)")

# Additional insights from covariates
print("\nEffects of other variables (from full model):")
for var in ['hours_studied', 'mother_edu', 'female']:
    if var in model_full.params:
        coef = model_full.params[var]
        pval = model_full.pvalues[var]
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var}: {coef:.4f} {sig} (p={pval:.4f})")

# Robustness check: local linear regression near cutoff
print("\n" + "="*80)
print("ROBUSTNESS CHECK: Local Linear Regression (Bandwidth = 0.5)")
print("="*80)

# Restrict to observations near cutoff
bandwidth = 0.5
local_data = df[abs(df['running_var_centered']) <= bandwidth].copy()

print(f"Observations within bandwidth: {len(local_data)} out of {len(df)}")

if len(local_data) > 20:  # Only if we have enough data
    model_local = ols(formula_full, data=local_data).fit()
    tau_local = model_local.params['D']
    
    print(f"\nLocal Treatment Effect (bandwidth={bandwidth}): {tau_local:.4f}")
    print(f"Standard Error: {model_local.bse['D']:.4f}")
    print(f"P-value: {model_local.pvalues['D']:.4f}")
    
    print(f"\nComparison with full sample estimate:")
    print(f"  Full sample: {tau_full:.4f}")
    print(f"  Local (bw={bandwidth}): {tau_local:.4f}")
    print(f"  Difference: {abs(tau_full - tau_local):.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  - continuity_test_plot.png")
print("  - rdd_discontinuity_plot.png")
