#  ben

## Model details
**Name:** ben 
**Project:** Biomarker Evaluation Node  
**Type:** binary probabilistic classifier  
**Reference implementation:** logistic regression with standardized numeric inputs

## Intended use
Research on software workflows for evaluating candidate canine cancer-signal models.

## Not intended for
- veterinary diagnosis
- treatment selection
- screening decisions
- replacing pathology, imaging, or veterinary judgment
- use on real patient data without an approved research protocol

## Inputs
The demo implementation uses synthetic fields that imitate a structured assay table. These names are placeholders and **do not represent validated canine cancer biomarkers**.

## Output
A probability in `[0, 1]` representing the model's estimated probability of the positive class within the experiment.

## Performance
No clinical performance is claimed. Demo metrics are based on synthetic data and only test the software path.

## Known limitations
- synthetic training/evaluation data
- no external validation
- no prospective validation
- no pathology-linked real-world cohort
- no validated threshold
- no regulatory review
