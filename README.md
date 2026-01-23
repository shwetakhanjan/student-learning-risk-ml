# student-learning-risk-ml
Predicting student learning risk using interpretable machine learning models

## Overview
This project investigates the application of machine learning to identify students who are at risk of falling behind in mathematics and to utilize data to determine the future courses of action via timely intervention. Based on my experience as a math educator, the goal is to model performance patterns that may indicate learning risk and form the basis of potential targeted interventions. This study is further supported by identifying two most important factors that determine the students at risk of falling behind. 
The project focuses on supervised learning techniques, model interpretability, and responsible evaluation. It prioritizes recall and error analysis over accuracy alone. In the effort to maximize the study, Recall gains a greater importance because we do not want to miss cases of false negatives.

## Problem Statement
In my experience as a math instructor, early detection of students at risk of falling behind in their math courses can go a long way in saving invaluable instructional resources and time, not to mention saving the student endless misery. The goal is to model performance patterns that may indicate learning risk and inform potential targeted interventions. I have modeled this problem as a binary classification task by predicting whether a student is at risk based on historical performance and engagement-related features.

## Dataset
The dataset represents student details and student-level academic performance as 'features' including:
-student_id 
 -age
-grade
-homework_completion
-practice_tests_avg
-practice completion rates
-class_participation
-geometry_score
-algebra1_score
-recent_quiz_avg
-study_hours_per_week 

The label/target is the ‘at_risk’ column that predicts whether the student is at risk of performing poorly. A label of ‘0’ means the student is not at risk, while those scoring ‘1’ are the ones that require timely intervention.  

I used synthetically generated data, keeping in mind real classroom learning patterns, historical performance, and engagement-related features. The code used for data generation can be found in the data exploration notebook. I validated the data through exploratory analysis to ensure the patterns and distributions made sense.


## Approach
The modeling pipeline includes:
- Data cleaning and exploratory analysis
- Feature engineering based on performance trends
- Baseline modeling using logistic regression
- Tree-based modeling using decision tree classifiers
- Model evaluation using precision, recall, and confusion matrices
I gave importance to the interpretability and evaluation trade-offs (particularly precision versus recall) based on situations in an educational set-up. I chose to go with Logistic Regression(LR) first as a baseline model due to its high interpretability and robustness. A high level of interpretability meant that each feature had a coefficient, and that I could explain how and why a particular feature affects risk. This was critical to my study as it helped in the next stage of intervention. The robustness of the model ensured that it performed reliably even with limited data and was less likely to overfit compared to a complex model. This reflected a stable behavior of the model on the synthetic dataset.
I used LR first to establish an interpretable and reliable baseline. I used the decision tree classifier after LR to capture a) non-linear relationships that might be missed by LR model and also b) feature interactions that a linear model like Logistic Regression cannot represent.

## Tools & Technologies
- Python
- pandas, numpy
- scikit-learn
- Jupyter notebooks
- GitHub 
## Project Workflow:
Synthetic Data → EDA → Feature Engineering → Model Training(Logistic Regression / Decision Tree) → Evaluation (ROC-AUC, Precision, Recall)

## Project Structure
- `data/`: raw and processed datasets
- `notebooks/`: exploratory analysis and modeling
- `results/`: evaluation outputs and metrics
Preprocessing and training code were kept in notebooks rather than refactored into a reusable src/ directory, which would be an important next step for production readiness.

## Modeling Approach

This project frames student learning risk as a binary classification problem. The goal is to identify students who may require early intervention based on engagement and performance features.

### Models
Two models were evaluated:

- Logistic Regression (baseline, interpretable)
- Decision Tree Classifier (non-linear model)

Logistic Regression served as a baseline model due to its interpretability and robustness. This allows clear reasoning about how the features influence the outcome or the ‘label’. LR helps in error tradeoffs before introducing a more flexible tree-based model. Logistic Regression lets me prioritize Recall over precision and that is a great plus, because it lets me accept controlled false positives at the cost of false negatives, and in this particular study false negatives are more costly than false positives. Missing an ‘at-risk’ student leads to no intervention while flagging a falsely negative student as positive only leads to a greater support for the student.

Once I had that baseline, I moved to a decision tree to see if a more flexible model could do better. Decision trees naturally model feature interactions, threshold-based decisions and non-linear patterns. I used the Decision Trees Classifier model to check if those patterns existed and whether the signal in the data is genuinely strong, or am I underfitting.


### Evaluation
Models were evaluated using:
- ROC-AUC
- Precision and recall (with emphasis on recall for at-risk students)
	Recall = (True Positive) / (True Positive+False Negative)
	Precision = (True Positive) / (True Positive+False Positive)
- Confusion matrices


### Results
Logistic regression achieved a ROC-AUC of 0.93. It  correctly identified most at-risk students but missed out some cases. The Decision Tree(DT) achieved higher performance with ROC-AUC: 0.98.  DT performed better across all metrics, including near-perfect recall of 0.96. This raised concerns about overfitting. It also reflected clean separability of the data, which is rare, but can be attributed to the synthetic data. It makes us skeptical of near perfect results and warrants us to tread with caution when evaluating results based on the Decision Tree Classifier model.


### Key Takeaway
This result leads us to compare between the two models and accept the Logistic Regression model over the Decision Tree model. The Decision Tree model is a clear case of overfitting  and this also shows clean separability of the data. Real world data may not highlight such clean separability. However, this fact highlights the importance of the trade-off between model interpretability and performance. We accept the Logistic Regression model and we need to go on from there to begin our work in the next level.


## Future Work
For future work, we need to experiment with ensemble methods like the Random Forest Classifier and can also use gradient boosting, leading to reduced overfitting by combining the strengths of multiple models. In future work, we can incorporate temporal features in the data, like making use of timestamps in the data related to past assessment. This might help us focus on understanding the context, capturing trends and cycles and growth patterns of students over the time that lead us to better predict and assess risk of falling behind and warrant timely intervention. We can also continue our work with considerations made for fairness and bias, objectivity of the tests and standardization.

## Author
Shweta Shroff

