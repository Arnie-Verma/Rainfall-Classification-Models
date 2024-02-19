# Predictive Modeling Project

## Project Overview
This project involves creating and analyzing a dataset related to humidity prediction for specific locations. The primary focus is on exploring the data, pre-processing, building classification models, evaluating model performance, and conducting investigative tasks.

### Data Creation
- Cleared the workspace and set the number of significant digits.
- Read data from "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package" into R.
- Created a subset of the data using random sampling for 10 locations and 2000 rows.

### Initial 
1. **Explored the Data:**
   - Analyzed the proportion of days with increasing/decreasing humidity.
   - Described predictor variables, including mean, standard deviations, etc.
   - Identified noteworthy attributes and considerations for omission.

2. **Pre-processing:**
   - Documented pre-processing steps required for model fitting.

3. **Data Split:**
   - Divided the data into 70% training and 30% test sets using a random seed.

4. **Classification Models:**
   - Implemented classification models for Decision Tree, Naïve Bayes, Bagging, Boosting, and Random Forest.

5. **Model Evaluation:**
   - Classified test cases and created a confusion matrix.
   - Reported accuracy for each model.

6. **ROC Curve:**
   - Calculated confidence and constructed ROC curves for each classifier.
   - Calculated AUC for each classifier.

7. **Results Comparison:**
   - Created a table comparing results from questions 5 and 6.
   - Identified if there's a single "best" classifier.

### Investigation
8. **Variable Importance:**
   - Determined the most important variables for predicting humidity changes.
   - Explored variables that could be omitted without significant performance impact.

9. **Simplified Classifier:**
   - Created a simplified classifier for manual classification.
   - Evaluated performance using test data and compared it to previous measures.

10. **Improved Tree-Based Classifier:**
    - Developed the best tree-based classifier by adjusting parameters or using cross-validation.
    - Compared the model's performance with others and explained decision-making.

11. **Artificial Neural Network (ANN):**
    - Implemented an ANN classifier.
    - Reported performance, commented on attributes, and discussed data pre-processing.

12. **SVM Classifier Implementation:**
    - Introduced SVM classifier.
    - Provided details on the model, package used, and its performance.

## Conclusion
This project demonstrates skills in data exploration, pre-processing, model building, evaluation, and investigative analysis. The variety of classifiers and the exploration of new models contribute to a comprehensive understanding of humidity prediction in the dataset.
