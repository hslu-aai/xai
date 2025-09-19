---
title: Home
layout: default
---

# HSLU I.BA_XAI Module

This material refers to the HSLU I.BA_XAI Module.

## Lecture notes

- [Lecture 1](xai_lecture_1.md)

## Weekly plan

### Week 1: Introduction to Explainable AI

- Module organization
- Motivation and definitions
- Goals and types of interpretability
- Overview and taxonomy

### Week 2: Interpretable Models

- Interpretation of linear and logistic regression
- Interpretation of decision trees and rule-based models
- Interpretation of nearest neighbours
<!--- Hands-on: Building and interpreting trees and rules -->

### Week 3: Feature Variations

- Ceteris Paribus
- Individual Conditional Expectations
- Partial Dependence Plots
- Accumulated Local Effects
<!--- Hands-on: Visualizations -->

### Week 4: Feature Importance

- Filter, wrapper, embedded methods
- Correlation and mutual information
- Permutation and leave-one-feature-out
<!--- Hands-on: Feature importance analysis -->

### Week 5: Feature Selection

- LASSO
- Best subset selection
- Forward selection, backward elimination
<!--- Implementation -->

### Week 6: LIME

- Local and global surrogates
- LIME
- LIME implementation details
- Visualizations of LIME
<!--- Hands-on: Applying LIME and SHAP to real models -->

### Week 7: SHAP

- Shapley values
- SHAP
- SHAP implementation details
- Visualizations of SHAP
<!--- Hands-on: Comparing PDP vs ICE vs SHAP -->

### Week 8: Practice

- Exercises on feature variations, importance, and selection
- Exercises on LIME and SHAP
- Critical assessment
<!--- Hands-on: Generate and evaluate counterfactuals -->

### Week 9: Prototypes and Counterfactuals

- Prototypes
- Counterfactuals and methods to generate them
- Anchors
<!--- Hands-on: Generate and evaluate counterfactuals -->

### Week 10: Neural Network Responses

- Feature visualization
- Neuron activations
- Attention as an explanation

### Week 11: Neural Network Gradients

- Gradient-based methods
- Saliency maps
<!--- Hands-on: Explain CNN predictions using saliency maps -->

### Week 12: Data and Latent Space

- Influence functions
- Concept vectors

### Week 13: Evaluation and Applications

- Metrics for explanation quality
- Human-centered evaluation
- Model monitoring and trust
- Bias detection and mitigation

### Week 14: Wrap-Up

- Future of explainable AI
- Review and exam preparation


## Resources

The three main books this module is based on are:

1. Molnar, C. (2025). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (3rd ed.). <https://christophm.github.io/interpretable-ml-book>
1. Di Cecco, A. & Gianfagna, L. (2025). Explainable AI with Python (2nd ed.). <https://doi.org/10.1007/978-3-031-92229-9>
1. Bhattacharya, A. (2022). Applied Machine Learning Explainability Techniques: Make ML models explainable and trustworthy for practical applications using LIME, SHAP, and more. [Packt Publishing](https://www.packtpub.com/en-cy/product/applied-machine-learning-explainability-techniques-9781803246154)

Additional resources for learning are given below.

### Books

- Mehta, M., Palade, V., & Chatterjee, I. (Eds.). (2022). Explainable AI: Foundations, Methodologies, and Applications (Vol. 232). Springer Nature. DOI: <https://doi.org/10.1007/978-3-031-12807-3>
- Lim, C. P., Vaidya, A., Jain, K., Mahorkar, V. U., & Jain, L. C. (2022). Handbook of artificial intelligence in healthcare. Springer International Publishing. DOI: <https://doi.org/10.1007/978-3-030-83620-7>
- Dignum, V. (2019). Responsible artificial intelligence: how to develop and use AI in a responsible way (p. 59). Cham: Springer. DOI: <https://doi.org/10.1007/978-3-030-30371-6>
- Samek, W., Montavon, G., Vedaldi, A., Hansen, L. K., & Müller, K. R. (Eds.). (2019). Explainable AI: interpreting, explaining, and visualizing deep learning (Vol. 11700). Springer Nature. DOI: <https://doi.org/10.1007/978-3-030-28954-6>

### Courses

- <https://interpretable-ml-class.github.io>
- <https://www.coursera.org/specializations/explainable-artificial-intelligence-xai>

### Awesome lists

- <https://github.com/pbiecek/xai_resources>
- <https://github.com/lopusz/awesome-interpretable-machine-learning>
- <https://github.com/altamiracorp/awesome-xai>
