# Adult Income Classification using Machine Learning Models and deploying to Streamlit

## 1. Problem Statement

The objective of this project is to predict whether an individual earns more than $50k or less than $50k per year based on some attributes such as sector they work (Workclass), education, marital status, relationship status, race, gender, hours per week and native country. This is a binary classification problem to be solved by implementing 6 machine learning models and deployed through an interactive Streamlit web application.



## 2. Dataset Description

The dataset used is the UCI Adult Income Dataset obtained from Kaggle (https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

### Key Characteristics:
- Total Instances: 45222 (after cleaning)
- Total Features: 14
- Target Variable: Income (>50K or <=50K)
- Feature Types: Numerical and Categorical

### Preprocessing Steps:
- Removed rows with ? values to avoid value errors for model stability and accurate learning
- Cleaned inconsistent income labels.



## 3. Machine Learning Models Implemented

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)



## 4. Model Performance Comparison

|        ML Model     |  Accuracy  |    AUC    |  Precision  |   Recall   |  F1 Score  |    MCC     |
|---------------------|------------|-----------|-------------|------------|------------|------------|
| Logistic Regression |   0.7999   |   0.8182  |   0.7856    |   0.7999   |   0.7787   |    0.408   |
| Decision Tree       |   0.8132   |   0.7567  |   0.8143    |   0.8132   |   0.8137   |    0.5103  |
| KNN                 |   0.7679   |   0.6858  |   0.7456    |   0.7679   |   0.7484   |    0.3157  |
| Naive Bayes         |   0.7889   |   0.8342  |   0.7719    |   0.7889   |   0.7612   |    0.3631  |
| Random Forest       |   0.8594   |   0.909   |   0.8545    |   0.8594   |   0.855    |    0.6121  |
| XGBoost             |   0.8729   |   0.9311  |   0.869     |   0.8729   |   0.869    |    0.6502  |



## 5. Model Performance Observations

|        Model        | Observation |
|---------------------|-------------|
| Logistic Regression | Performs well as a baseline model with stable and interpretable results but struggles with nonlinear patterns. |
| Decision Tree       | Captures complex relationships but shows signs of overfitting on training data. |
| KNN                 | Performance depends heavily on distance calculations and scales poorly with larger datasets. |
| Naive Bayes         | Fast and efficient but assumes feature independence which limits accuracy. |
| Random Forest       | Provides strong performance due to ensemble learning and reduces overfitting compared to single trees. |
| XGBoost             | Achieves the best overall performance by boosting weak learners and optimizing loss efficiently. |



## 6. Streamlit Application Features

The deployed Streamlit application includes:

- CSV dataset upload functionality  
- Machine learning model selection dropdown  
- Display of evaluation metrics:
  - Accuracy
  - AUC Score
  - Precision
  - Recall
  - F1 Score
  - MCC Score  
- Confusion matrix visualization  
- Full comparison table of all models 
