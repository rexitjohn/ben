# Biomarker Evaluation Node (ben)

**ben** is an experimental veterinary machine-learning research project for evaluating whether structured biomarker-like signal patterns can help distinguish **canine cancer-positive** samples from **benign/control** samples.

> **Research use only. Not a veterinary diagnostic device.**  
> This repository does not contain validated clinical claims or real patient records. The included demo pipeline uses **synthetic data** so the software can be tested safely and reproducibly.

## Why ben exists

ben packages preprocessing, model training, calibration-minded evaluation, and experiment reporting into a small auditable pipeline. The project emphasizes reproducibility, holdout testing, sensitivity/specificity reporting, subgroup checks, data-leakage prevention, and clear limitations.

## Current status

**Alpha 1.0 - research prototype**

Included today:
- synthetic canine assay-data generator
- preprocessing pipeline
- logistic-regression baseline
- stratified holdout evaluation
- ROC-AUC, sensitivity, specificity, precision, NPV, F1 and Brier score
- confusion-matrix and ROC plots
- model card and validation notes
- automated tests and CI

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/ben.git
cd ben
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ben train --config configs/alpha_1_0.yaml
```

Outputs are written to `artifacts/alpha_1_0/`.

## Repository layout

```text
ben/
├── src/ben/                  Python package
├── configs/                  experiment configuration
├── experiments/              versioned experiment notes
├── docs/                     methodology and governance
├── tests/                    automated tests
├── data/                     documentation only; no private records
├── assets/figures/           generated figures
└── .github/                  CI and contribution templates
```

## Data policy

Do **not** commit veterinary records, owner details, clinic identifiers, raw pathology reports, or other sensitive data.

Generate synthetic demo data locally with:

```bash
ben make-demo-data --rows 1000 --output data/demo_synthetic.csv
```

## Model outputs

ben produces **research probabilities**, not diagnoses. A probability threshold used during experiments is a modeling choice and should not be treated as a clinical cutoff without prospective validation.

## Validation philosophy

Any future claim of clinical utility should require, at minimum: locked preprocessing/model weights, external-site validation, pre-specified endpoints and thresholds, subgroup analysis, confidence intervals, veterinary/statistical review, and prospective evaluation.

See [`docs/validation.md`](docs/validation.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
