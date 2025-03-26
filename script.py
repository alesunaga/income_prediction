import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean whitespace from categorical columns
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df['income'].head())

# Create feature dataframe X with dummy variables
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education']
X = pd.get_dummies(df[feature_cols], drop_first=True)

# Plot feature correlation heatmap
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Create binary target variable y
y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)

# Print model parameters
print("Intercept:", log_reg.intercept_)
print("Coefficients:", log_reg.coef_)

# Evaluate the model
print('Confusion Matrix on test set:')
print(confusion_matrix(y_test, y_pred))

print('Accuracy Score on test set:')
print(accuracy_score(y_test, y_pred))

# Create DataFrame of coefficients and variable names
coefficients = log_reg.coef_[0]
variable_names = X.columns
coeff_df = pd.DataFrame(list(zip(variable_names, coefficients)), columns=['Variable', 'Coefficient'])
coeff_df = coeff_df[coeff_df['Coefficient'] != 0].sort_values(by='Coefficient')
print(coeff_df)

# Plot the coefficients
plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Variable', data=coeff_df)
plt.title('Logistic Regression Coefficients')
plt.show()

# Plot the ROC curve and print the AUC
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print('AUC:', roc_auc)
