#!/usr/bin/env python3
"""
Main script to run both DiD and RDD analyses
Usage: python main.py [--did] [--rdd]
  If no arguments provided, runs both analyses
"""

import sys
import os

def run_did_analysis():
    """Run Difference-in-Difference analysis"""
    print("\n" + "#"*80)
    print("# RUNNING DIFFERENCE-IN-DIFFERENCE (DiD) ANALYSIS")
    print("#"*80 + "\n")
    
    # Import and run the DiD analysis
    exec(open('did_analysis.py').read())
    
def run_rdd_analysis():
    """Run Regression Discontinuity Design analysis"""
    print("\n" + "#"*80)
    print("# RUNNING REGRESSION DISCONTINUITY DESIGN (RDD) ANALYSIS")
    print("#"*80 + "\n")
    
    # Import and run the RDD analysis
    exec(open('rdd_analysis.py').read())

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if '--did' in sys.argv:
            run_did_analysis()
        if '--rdd' in sys.argv:
            run_rdd_analysis()
    else:
        # Run both if no arguments specified
        run_did_analysis()
        print("\n\n")
        run_rdd_analysis()
    
    print("\n" + "#"*80)
    print("# ALL ANALYSES COMPLETE")
    print("#"*80)
