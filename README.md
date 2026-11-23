# Fine-Tuned Transformers for Requirements Classification

Requirements engineers spend substantial manual effort separating functional requirements from non-functional ones and tagging each non-functional requirement with its quality type, such as security, performance, or usability. On large programs this triage is slow, varies between reviewers, and is hard to keep traceable as documents change.

This project asks whether a fine-tuned transformer classifies requirements more accurately than a strong classical baseline, where any advantage comes from, and whether it holds on projects the models have never seen.

## Approach

Fine-tuned DistilBERT is compared against TF-IDF with class weighted logistic regression on the PROMISE_exp dataset of 968 requirements from 47 projects. There are two tasks, functional versus non-functional, and classification into eleven non-functional quality types. Both models are tuned and scored on identical repeated cross validation folds, on a held out test set, and on 15 folds in which whole projects are held out. Macro F1 is the headline metric, and corrected resampled t-tests decide four hypotheses. An ablation and a learning curve explain where the differences come from.

## Findings

| Setting | Task | Baseline macro F1 | DistilBERT macro F1 |
|---|---|---|---|
| Within projects | Functional vs non-functional | 0.900 | 0.922 |
| Within projects | Eleven quality types | 0.606 | 0.642 |

- Within projects the transformer's gains are small and, averaged over three training seeds, not statistically significant. They are also not concentrated in the rare quality types.
- On projects held out from training, both models degrade but the baseline degrades more. The transformer's advantage on the eleven class task at least doubles to 0.085 macro F1, is significant, and holds on all 15 project grouped folds.
- Masking project specific vocabulary affects neither model, which points to differences in how projects phrase and label requirements rather than to project names.
- The baseline depends on class weighting far more than the transformer does, and the transformer overtakes the baseline only beyond about 15 labeled examples per class.

The value of pretraining for requirements classification lies mainly in robustness to unseen projects rather than in accuracy on familiar ones.

## Status

The research notebook, results, and IEEE format paper will be published here after the Fall 2026 term ends.

## Built with

Python, PyTorch, Hugging Face Transformers, scikit-learn, SciPy

## Context

Research project for EN.705.742 Advanced Applied Machine Learning at Johns Hopkins University (Fall 2026).
