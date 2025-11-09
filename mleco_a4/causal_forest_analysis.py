#!/usr/bin/env python3
"""
Causal Forest Analysis for Health Program Heterogeneous Treatment Effects
Estimates both Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleCausalForest:
    """
    Simplified Causal Forest implementation using Random Forest
    Based on the T-learner approach
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_control = None
        self.model_treated = None
        
    def fit(self, X, y, treatment):
        """
        Fit separate models for treatment and control groups
        """
        control_mask = treatment == 0
        treated_mask = treatment == 1
        
        self.model_control = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=10,
            min_samples_leaf=20
        )
        self.model_treated = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=10,
            min_samples_leaf=20
        )
        
        self.model_control.fit(X[control_mask], y[control_mask])
        self.model_treated.fit(X[treated_mask], y[treated_mask])
        
        return self
    
    def predict_cate(self, X):
        """
        Predict Conditional Average Treatment Effect
        CATE(x) = E[Y(1)|X=x] - E[Y(0)|X=x]
        """
        y1_pred = self.model_treated.predict(X)
        y0_pred = self.model_control.predict(X)
        return y1_pred - y0_pred
    
    def estimate_ate(self, X):
        """
        Estimate Average Treatment Effect as the mean CATE
        """
        cate = self.predict_cate(X)
        return np.mean(cate), np.std(cate) / np.sqrt(len(cate))

def run_causal_forest_analysis():
    """Run complete Causal Forest analysis"""
    
    print("="*80)
    print("CAUSAL FOREST ANALYSIS")
    print("="*80)
    
    df = pd.read_csv('sim_health.csv')
    print(f"\nDataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    
    Y = df['Y'].values
    D = df['D'].values
    
    covariate_cols = [col for col in df.columns if col not in ['Y', 'D']]
    X_df = df[covariate_cols].copy()
    
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Converting categorical variables to dummies: {categorical_cols}")
        X_df_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    else:
        X_df_encoded = X_df
    
    X = X_df_encoded.values
    
    print(f"Treatment (D): {D.sum()} treated, {len(D) - D.sum()} control")
    print(f"Covariates: {X.shape[1]} variables")
    
    print("\n" + "-"*80)
    print("Fitting Causal Forest (T-Learner with Random Forest)")
    print("-"*80)
    
    cf = SimpleCausalForest(n_estimators=200, random_state=42)
    cf.fit(X, Y, D)
    
    ate, ate_se = cf.estimate_ate(X)
    
    print(f"\nAverage Treatment Effect (ATE): {ate:.4f}")
    print(f"Standard Error: {ate_se:.4f}")
    print(f"95% Confidence Interval: [{ate - 1.96*ate_se:.4f}, {ate + 1.96*ate_se:.4f}]")
    
    cate = cf.predict_cate(X)
    
    print(f"\nConditional Average Treatment Effect (CATE) Statistics:")
    print(f"  Mean: {np.mean(cate):.4f}")
    print(f"  Std: {np.std(cate):.4f}")
    print(f"  Min: {np.min(cate):.4f}")
    print(f"  Max: {np.max(cate):.4f}")
    
    print("\n" + "-"*80)
    print("Analyzing Treatment Effect Heterogeneity")
    print("-"*80)
    
    df['cate'] = cate
    
    print("\n1. CATE by Region:")
    region_cate = df.groupby('region')['cate'].agg(['mean', 'std', 'count'])
    print(region_cate)
    
    plt.figure(figsize=(10, 6))
    region_means = df.groupby('region')['cate'].mean().sort_values()
    region_std = df.groupby('region')['cate'].std()
    
    plt.barh(range(len(region_means)), region_means.values, 
             xerr=region_std.loc[region_means.index].values, 
             color='steelblue', alpha=0.7, capsize=5)
    plt.yticks(range(len(region_means)), region_means.index)
    plt.xlabel('Conditional Average Treatment Effect (CATE)', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.title('Treatment Effect Heterogeneity by Region', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cate_by_region.png', dpi=300, bbox_inches='tight')
    print("  Plot saved: cate_by_region.png")
    plt.close()
    
    print("\n2. CATE by Household Income Quartiles:")
    df['income_quartile'] = pd.qcut(df['household_income_usd'], q=4, 
                                     labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    income_cate = df.groupby('income_quartile')['cate'].agg(['mean', 'std', 'count'])
    print(income_cate)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    income_means = df.groupby('income_quartile')['cate'].mean()
    income_std = df.groupby('income_quartile')['cate'].std()
    
    axes[0].bar(range(len(income_means)), income_means.values,
                yerr=income_std.values, color='forestgreen', alpha=0.7, capsize=5)
    axes[0].set_xticks(range(len(income_means)))
    axes[0].set_xticklabels(income_means.index, rotation=0)
    axes[0].set_ylabel('Conditional Average Treatment Effect (CATE)', fontsize=11)
    axes[0].set_xlabel('Household Income Quartile', fontsize=11)
    axes[0].set_title('CATE by Income Quartiles (Bar Plot)', fontsize=12, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    
    df['income_binned'] = pd.cut(df['household_income_usd'], bins=20)
    income_bin_cate = df.groupby('income_binned')['cate'].mean()
    income_bin_centers = [interval.mid for interval in income_bin_cate.index]
    
    axes[1].scatter(df['household_income_usd'], df['cate'], alpha=0.3, s=10, color='gray')
    axes[1].plot(income_bin_centers, income_bin_cate.values, color='forestgreen', 
                linewidth=2, label='Binned Average CATE')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Effect')
    axes[1].set_xlabel('Household Income (USD)', fontsize=11)
    axes[1].set_ylabel('Conditional Average Treatment Effect (CATE)', fontsize=11)
    axes[1].set_title('CATE vs Household Income (Continuous)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cate_by_income.png', dpi=300, bbox_inches='tight')
    print("  Plot saved: cate_by_income.png")
    plt.close()
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"The Causal Forest estimates an Average Treatment Effect of {ate:.4f},")
    print(f"which measures the overall impact of the health program.")
    print(f"\nTreatment effects vary substantially across subgroups:")
    print(f"  - By region: from {region_cate['mean'].min():.4f} to {region_cate['mean'].max():.4f}")
    print(f"  - By income quartile: from {income_cate['mean'].min():.4f} to {income_cate['mean'].max():.4f}")
    print(f"\nThis heterogeneity suggests the program may be more effective for certain populations.")
    
    results = {
        'ate': ate,
        'ate_se': ate_se,
        'cate_mean': np.mean(cate),
        'cate_std': np.std(cate),
        'cate_min': np.min(cate),
        'cate_max': np.max(cate),
        'region_cate': region_cate,
        'income_cate': income_cate
    }
    
    return results

if __name__ == "__main__":
    results = run_causal_forest_analysis()
