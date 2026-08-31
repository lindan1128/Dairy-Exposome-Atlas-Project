# Dairy Exposome Atlas Code

This folder contains the manuscript-facing analysis code for the US dairy exposome atlas. Scripts are organized by study point and follow the Methods in `papers/NS/MS.docx`.

## Structure

- `1/`: endpoint ExWAS for total milk production and milk production per cow.
- `2/`: yearly exposure-association trends for milk production per cow.
- `3/`: rolling-origin milk-per-cow forecasting, SHAP attribution and random-forest robustness.
- `4/`: state-class exposure priority indices and maps.
- `lib_statistics_panel.py`: shared panel assembly and fixed-effect regression utilities.

## Methods Implemented

The endpoint ExWAS uses log outcomes, standardized numeric exposures, state and year-month fixed effects, milk-cow-inventory weights, state-clustered uncertainty and BY-FDR/Bonferroni correction across 204 curated exposures. Sensitivity and robustness analyses include state-specific trends, unweighted models, leave-one-state-out refitting, GAM, GEE and residualized rexposome ExWAS.

The yearly analysis estimates exposure-by-year standardized beta coefficients for milk production per cow, with sensitivity/robustness checks using FE + year-month, FE + exposure-by-year trend, LOSO, GAM, GEE and LME models.

Forecasting compares history-only and exposome-enhanced rolling-origin models for 1- to 12-month-ahead milk production per cow. Priority maps combine recent standardized beta, adjusted incremental R2 and 1- to 9-month SHAP attribution into state-normalized class priority shares.

## Environment

Python and R package versions used for manuscript analyses are listed in `requirements.txt` and `R-packages.txt`.
