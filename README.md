# student-learning-risk-ml
Predicting student learning risk and recommending targeted math practice using machine learning

# Predicting Student Learning Risk Using Machine Learning

## Overview
This project explores the use of machine learning to identify students at risk of falling behind in mathematics and to support targeted intervention through data-driven insights. Drawing from my experience as a math educator, the goal is to model performance patterns that may indicate learning risk and inform personalized practice recommendations.

The project focuses on supervised learning techniques, interpretability, and responsible model evaluation rather than maximizing accuracy alone.

## Problem Statement
In educational settings, students often struggle for extended periods before intervention occurs. Early identification of learning risk can enable timely, targeted support.  
This project frames the problem as a binary classification task: predicting whether a student is at risk based on historical performance and engagement-related features.

## Dataset
The dataset represents student-level academic performance, including:
- assessment scores
- practice completion rates
- time spent on exercises
- historical performance trends

The data used is synthetically generated to reflect realistic educational patterns.

## Approach
The modeling pipeline includes:
- Data cleaning and exploratory analysis
- Feature engineering based on performance trends
- Baseline modeling using logistic regression
- Tree-based modeling using decision tree classifiers
- Model evaluation using precision, recall, and confusion matrices

Interpretability and evaluation tradeoffs are prioritized to ensure responsible use in real-world contexts.

## Tools & Technologies
- Python
- pandas, numpy
- scikit-learn
- Jupyter notebooks
- GitHub for version control

## Project Structure
- `data/`: raw and processed datasets
- `notebooks/`: exploratory analysis and modeling
- `src/`: reusable preprocessing and training scripts
- `results/`: evaluation outputs and metrics

## Future Work
- Experiment with ensemble methods (e.g., Random Forest)
- Incorporate temporal features
- Explore fairness and bias considerations
- Extend recommendations based on predicted risk levels

## Author
Shweta Shroff

