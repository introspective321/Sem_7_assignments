# Sem_7_assignments

This repository contains implementations of causal inference methods:
1. **Difference-in-Difference (DiD) Analysis** - Estimating the causal impact of government subsidies on average wages
2. **Regression Discontinuity Design (RDD) Analysis** - Estimating the causal effect of scholarships on test scores

## Files

- `did_data.csv` - Dataset for Difference-in-Difference analysis
- `rdd_data.csv` - Dataset for Regression Discontinuity analysis
- `did_analysis.py` - Complete DiD analysis implementation
- `rdd_analysis.py` - Complete RDD analysis implementation
- `main.py` - Main script to run both analyses
- `requirements.txt` - Python package dependencies

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Run Both Analyses

```bash
python main.py
```

### Run Only DiD Analysis

```bash
python did_analysis.py
```

or

```bash
python main.py --did
```

### Run Only RDD Analysis

```bash
python rdd_analysis.py
```

or

```bash
python main.py --rdd
```

## Difference-in-Difference Analysis

The DiD analysis (`did_analysis.py`) performs the following:

**a) Creates 'treated' variable**: Identifies regions that receive government subsidy

**b) Creates 'post' variable**: Identifies time periods (2010+) after treatment begins

**c) Parallel trends plot**: Visualizes average wages over time for treated vs control groups to assess the parallel trends assumption (saves as `parallel_trends_plot.png`)

**d) Basic DiD regression**: Estimates the causal effect using the basic DiD model:
```
avg_wage = β₀ + β₁*treated + β₂*post + β₃*(treated*post) + ε
```
where β₃ is the DiD estimator

**e) DiD with controls**: Re-estimates the causal effect including control variables:
- population
- unemployment_rate
- gdp_per_capita
- exports_per_capita
- fdi_inflow

**f) Heterogeneous treatment effects**: Estimates the DiD separately for three sectors:
- Agriculture
- Manufacturing
- Services

Generates `heterogeneous_effects_plot.png` showing variation in treatment effects across sectors.

## Regression Discontinuity Analysis

The RDD analysis (`rdd_analysis.py`) performs the following:

**a) Creates treatment variable D**: Students with 5th_score > 0 receive scholarship (D=1), others don't (D=0)

**b) Continuity tests**: Plots `hours_studied` and `mother_edu` against `5th_score` to verify no manipulation at the cutoff (saves as `continuity_test_plot.png`)

**c) Discontinuity visualization**: Plots `10th_score` vs `5th_score` to illustrate the treatment effect (saves as `rdd_discontinuity_plot.png`)

**d) RDD estimation**: Estimates three models:
1. Basic RDD model
2. RDD with different slopes on each side of cutoff
3. Full RDD model with covariates (hours_studied, mother_edu, female)

Includes robustness checks using local linear regression with bandwidth = 0.5

## Outputs

### Plots Generated

1. `parallel_trends_plot.png` - Shows average wages over time for treated vs control groups
2. `heterogeneous_effects_plot.png` - Shows treatment effect variation across sectors
3. `continuity_test_plot.png` - Validates RDD assumptions (no manipulation)
4. `rdd_discontinuity_plot.png` - Shows the discontinuity in outcome at cutoff

### Statistical Results

Both scripts print detailed statistical results including:
- Regression coefficients and standard errors
- P-values and significance tests
- Model diagnostics (R-squared, etc.)
- Interpretations of causal estimates

## Methodology

### Difference-in-Difference
DiD estimates causal effects by comparing changes over time between treated and control groups. The key assumption is **parallel trends**: in the absence of treatment, both groups would have followed similar trajectories.

### Regression Discontinuity
RDD exploits a cutoff rule for treatment assignment. Students just above and below the cutoff are assumed to be similar except for treatment status, allowing causal inference. The key assumptions are:
- No manipulation of the running variable (5th_score)
- Continuity of potential outcomes at the cutoff