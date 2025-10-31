# Analysis Summary

## Difference-in-Difference (DiD) Analysis

### Purpose
Estimate the causal impact of government subsidies on average wages in different economic regions.

### Key Findings

#### Part A - Treated Variable
- Created binary variable indicating regions that ever receive treatment
- 100 treated regions and 100 control regions
- Balanced design with 1000 observations each

#### Part B - Post Variable
- Pre-treatment period: 2006-2009 (4 years)
- Post-treatment period: 2010-2015 (6 years)
- Treatment begins in 2010

#### Part C - Parallel Trends
- Visualized average wages over time for both groups
- Pre-treatment trends appear reasonably parallel
- Both groups show similar trajectories from 2006-2009
- Suggests the parallel trends assumption is plausible

#### Part D - Basic DiD Regression
- **DiD Estimate (β3): 1.7944**
- Standard Error: 0.1880
- P-value: < 0.001 (highly significant)
- **Interpretation**: The subsidy increases average wages by approximately **$1.79**

#### Part E - DiD with Control Variables
- **DiD Estimate with controls: 1.6834**
- Standard Error: 0.1942
- P-value: < 0.001 (highly significant)
- Controls included: population, unemployment_rate, gdp_per_capita, exports_per_capita, fdi_inflow
- **Interpretation**: After controlling for confounders, the subsidy increases wages by **$1.68**
- Estimate is robust and similar to basic model

#### Part F - Heterogeneous Treatment Effects
Treatment effects vary significantly across sectors:

| Sector | Effect | Std Error | P-value | Significant |
|--------|--------|-----------|---------|-------------|
| Manufacturing | **2.158** | 0.327 | < 0.001 | Yes |
| Agriculture | **1.509** | 0.425 | < 0.001 | Yes |
| Services | **1.456** | 0.299 | < 0.001 | Yes |

**Key Insights:**
- Manufacturing sector benefits most from the subsidy (+$2.16)
- Services sector benefits least (+$1.46)
- All three sectors show significant positive effects
- Variation range: $0.70 between highest and lowest

---

## Regression Discontinuity Design (RDD) Analysis

### Purpose
Estimate the causal effect of receiving a scholarship (for students scoring > 0 on normalized 5th grade test) on 10th grade test scores.

### Key Findings

#### Part A - Treatment Variable
- Students with 5th_score > 0 receive scholarship (D=1)
- Students with 5th_score ≤ 0 do not receive scholarship (D=0)
- 2,025 students received scholarship (50.6%)
- 1,975 students did not receive scholarship (49.4%)
- Nearly balanced design at cutoff

#### Part B - Continuity Tests
Tests whether covariates are continuous at the cutoff (would suggest no manipulation):

**Hours Studied:**
- Mean just below cutoff: 11.52
- Mean just above cutoff: 12.65
- P-value: < 0.001 (significant difference)

**Mother's Education:**
- Mean just below cutoff: 9.79 years
- Mean just above cutoff: 10.09 years
- P-value: 0.0035 (significant difference)

⚠️ **Warning**: Both covariates show discontinuity at the cutoff, which may suggest some manipulation or selection. This violates RDD assumptions and means results should be interpreted with caution.

#### Part C - Discontinuity Visualization
- Visual inspection shows a jump in 10th grade scores at the cutoff
- Linear fit shows discontinuity of approximately 0.19 points
- Scatter plot clearly shows treated vs untreated students

#### Part D - RDD Estimation

Three models estimated:

| Model | Treatment Effect | Std Error | P-value | Significant |
|-------|------------------|-----------|---------|-------------|
| Basic RDD | 0.362 | 0.429 | 0.399 | No |
| RDD with Interaction | 0.188 | 0.384 | 0.624 | No |
| **RDD with Covariates** | **0.152** | 0.383 | 0.692 | **No** |

**Full Model Interpretation:**
- Treatment effect estimate: 0.15 points increase in 10th grade score
- P-value: 0.692 (NOT statistically significant at 5% level)
- **Conclusion**: Cannot conclude that scholarship has a causal effect on 10th grade scores

**Other significant predictors:**
- Hours studied: +0.237 points per additional hour (p < 0.001) ***
- Mother's education: +0.045 points per year (p = 0.437) - not significant
- Female: +0.079 points (p = 0.734) - not significant

#### Robustness Check (Local Linear Regression)
- Using bandwidth = 0.5 around cutoff (1,552 observations)
- Local treatment effect: **6.13 points** (p < 0.001)
- This is very different from full sample estimate (0.15)
- Suggests effect is stronger near the cutoff
- Large discrepancy indicates sensitivity to specification

---

## Generated Plots

1. **parallel_trends_plot.png** - Shows wage trends over time for treated vs control regions
2. **heterogeneous_effects_plot.png** - Bar chart comparing treatment effects across sectors
3. **continuity_test_plot.png** - Tests continuity of covariates at RDD cutoff (hours_studied and mother_edu)
4. **rdd_discontinuity_plot.png** - Visualizes discontinuity in 10th grade scores at cutoff

---

## How to Run

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run both analyses:
```bash
python main.py
```

### Run only DiD:
```bash
python did_analysis.py
# or
python main.py --did
```

### Run only RDD:
```bash
python rdd_analysis.py
# or
python main.py --rdd
```

---

## Statistical Methods Used

### Difference-in-Difference
- OLS regression with treatment, post, and interaction terms
- Standard errors assume homoskedasticity
- Relies on parallel trends assumption

### Regression Discontinuity
- Sharp RDD design (deterministic treatment assignment)
- Linear regression with running variable
- Allows different slopes on each side of cutoff
- Includes robustness checks with local linear regression

---

## Conclusions

### DiD Analysis
✅ **Strong evidence** that government subsidies have a positive causal effect on wages
- Effect size: ~$1.68-1.79 increase
- Highly statistically significant
- Robust to inclusion of control variables
- Heterogeneous effects across sectors with Manufacturing benefiting most

### RDD Analysis
❌ **Weak evidence** for causal effect of scholarship on test scores
- Small, non-significant effect in full sample (0.15 points)
- Evidence of potential manipulation at cutoff (covariates discontinuous)
- Results sensitive to specification (local vs global estimation)
- Hours studied is a stronger predictor than scholarship receipt
- RDD assumptions may be violated
