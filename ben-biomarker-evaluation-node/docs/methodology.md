# Methodology

## Objective
Evaluate whether the software can support reproducible binary-classification experiments for canine cancer-signal research.

## Alpha 1.0 baseline
- stratified holdout split
- median imputation
- feature standardization
- class-balanced logistic regression
- probability output
- fixed experiment threshold
- ROC-AUC and threshold metrics

Each experiment should record repository commit, data version, random seed, sample counts, feature schema, preprocessing, model parameters, decision threshold, metrics, uncertainty, and limitations.
