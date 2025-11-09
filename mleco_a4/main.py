#!/usr/bin/env python3
"""
Main script to run both Double-LASSO and Causal Forest analyses
Usage: python main.py
"""

import sys
import subprocess

def run_double_lasso():
    """Run Double-LASSO (DML) analysis"""
    print("\n" + "#"*80)
    print("# RUNNING DOUBLE-LASSO (DML) ANALYSIS")
    print("#"*80 + "\n")
    
    subprocess.run([sys.executable, 'double_lasso_analysis.py'])
    
def run_causal_forest():
    """Run Causal Forest analysis"""
    print("\n" + "#"*80)
    print("# RUNNING CAUSAL FOREST ANALYSIS")
    print("#"*80 + "\n")
    
    subprocess.run([sys.executable, 'causal_forest_analysis.py'])

def compare_results():
    """Compare ATE from both methods"""
    print("\n" + "#"*80)
    print("# COMPARING DOUBLE-LASSO AND CAUSAL FOREST RESULTS")
    print("#"*80 + "\n")
    
    print("Both methods estimate the Average Treatment Effect (ATE) of the health program:")
    print("  - Double-LASSO: Uses debiased machine learning with LASSO for covariate selection")
    print("  - Causal Forest: Uses random forests to model treatment and control outcomes separately")
    print("\nThe two methods provide complementary perspectives on causal effects:")
    print("  - Double-LASSO gives a single ATE with rigorous statistical inference")
    print("  - Causal Forest provides ATE plus heterogeneous effects (CATE) across subgroups")
    print("\nRefer to the generated plots and README.md for detailed results.")

if __name__ == "__main__":
    run_double_lasso()
    print("\n\n")
    run_causal_forest()
    print("\n\n")
    compare_results()
    
    print("\n" + "#"*80)
    print("# ALL ANALYSES COMPLETE")
    print("#"*80)
    print("\nGenerated outputs:")
    print("  - cate_by_region.png")
    print("  - cate_by_income.png")
    print("  - See README.md for complete results summary")
