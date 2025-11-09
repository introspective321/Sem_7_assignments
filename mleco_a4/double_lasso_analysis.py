#!/usr/bin/env python3
"""
Double-LASSO (DML) Analysis for Health Program Causal Effect
Estimates the Average Treatment Effect of health program participation on health outcomes
using Double/Debiased Machine Learning with cross-fitting
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def double_lasso_dml(Y, D, X, n_folds=5, random_state=42):
    """
    Double/Debiased LASSO with cross-fitting
    
    Parameters:
    Y: outcome variable
    D: treatment variable
    X: covariates
    n_folds: number of folds for cross-fitting
    random_state: random seed
    
    Returns:
    theta: ATE estimate
    se: standard error
    """
    n = len(Y)
    theta_folds = []
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    residuals_Y = np.zeros(n)
    residuals_D = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        
        lasso_Y = LassoCV(cv=5, random_state=random_state, max_iter=5000)
        lasso_Y.fit(X_train, Y_train)
        Y_pred = lasso_Y.predict(X_test)
        residuals_Y[test_idx] = Y_test - Y_pred
        
        lasso_D = LassoCV(cv=5, random_state=random_state, max_iter=5000)
        lasso_D.fit(X_train, D_train)
        D_pred = lasso_D.predict(X_test)
        residuals_D[test_idx] = D_test - D_pred
    
    theta = np.sum(residuals_Y * residuals_D) / np.sum(residuals_D ** 2)
    
    residuals_final = residuals_Y - theta * residuals_D
    variance = np.mean(residuals_final ** 2) / np.mean(residuals_D ** 2)
    se = np.sqrt(variance / n)
    
    return theta, se

def run_double_lasso_analysis():
    """Run complete Double-LASSO analysis"""
    
    print("="*80)
    print("DOUBLE-LASSO (DML) ANALYSIS")
    print("="*80)
    
    df = pd.read_csv('sim_health.csv')
    print(f"\nDataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    
    Y = df['Y'].values
    D = df['D'].values
    
    covariate_cols = [col for col in df.columns if col not in ['Y', 'D']]
    X_df = df[covariate_cols]
    
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Converting categorical variables to dummies: {categorical_cols}")
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    
    X = X_df.values
    
    print(f"Treatment (D): {D.sum()} treated, {len(D) - D.sum()} control")
    print(f"Covariates: {X.shape[1]} variables (after encoding)")
    print(f"High-dimensional setting: {X.shape[1]} covariates for {len(D)} observations")
    
    print("\n" + "-"*80)
    print("Estimating Average Treatment Effect using Double-LASSO with Cross-Fitting")
    print("-"*80)
    
    theta, se = double_lasso_dml(Y, D, X, n_folds=5, random_state=42)
    
    t_stat = theta / se
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(len(Y)))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se
    
    print(f"\nAverage Treatment Effect (ATE): {theta:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if abs(t_stat) > 1.96:
        print(f"Result: Statistically significant at 5% level (|t| > 1.96)")
    else:
        print(f"Result: Not statistically significant at 5% level (|t| <= 1.96)")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"The health program participation has an average causal effect of {theta:.4f} on health outcomes.")
    print(f"This is the estimated difference in health outcome between participants and non-participants,")
    print(f"controlling for all {X.shape[1]} pre-treatment covariates using Double-LASSO.")
    
    results = {
        'ate': theta,
        'se': se,
        't_stat': t_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': len(Y),
        'n_treated': D.sum(),
        'n_control': len(D) - D.sum(),
        'n_covariates': X.shape[1]
    }
    
    return results

if __name__ == "__main__":
    results = run_double_lasso_analysis()
