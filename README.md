Logistic Regression for Income Prediction

This script performs logistic regression to predict income levels (>50K or <=50K) 
using the Adult Census Income dataset. It includes data preprocessing, model training, 
evaluation, and visualization of results.

Dataset:
    Adult Census Income dataset (adult.data)

Features:
    'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education'

Target Variable:
    'income' (binary: 0 for <=50K, 1 for >50K)

Steps:
    1. Data Loading and Cleaning:
        - Loads the dataset from 'adult.data'.
        - Cleans whitespace from categorical columns.
    2. Feature Engineering:
        - Creates dummy variables for categorical features using pd.get_dummies().
        - Drops the first dummy variable to avoid multicollinearity.
    3. Correlation Analysis:
        - Generates a heatmap of feature correlations using seaborn.
    4. Target Variable Creation:
        - Creates a binary target variable 'y' from the 'income' column.
    5. Data Splitting:
        - Splits the data into training and testing sets using train_test_split().
    6. Model Training:
        - Trains a Logistic Regression model with L1 regularization.
    7. Model Evaluation:
        - Prints the model's intercept and coefficients.
        - Prints the confusion matrix and accuracy score.
        - Creates a DataFrame of coefficients and variable names.
        - Generates a bar plot of the coefficients.
        - Plots the ROC curve and prints the AUC value.
