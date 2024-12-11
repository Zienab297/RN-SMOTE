# Reduced Noise SMOTE

This repository contains a Jupyter Notebook for analyzing customer churn data and building a predictive model. The project demonstrates key steps in data preprocessing, exploratory data analysis (EDA), and machine learning model development.

## Project Overview

Customer churn is the rate at which customers stop doing business with a company. Predicting and reducing churn is vital for maintaining a stable customer base. This notebook follows these steps:

1. **Data Import and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Model Building and Evaluation**
4. **Reduced-Noise Synthetic Minority Oversampling Technique (RN-SMOTE)**

## Notebook Contents

### 1. Data Import and Preprocessing

The notebook starts by importing the dataset and performing essential preprocessing tasks such as:

- **Handling Missing Values:** Imputation methods are applied to address missing entries.
- **Feature Encoding:** Categorical variables are transformed using one-hot or label encoding.
- **Feature Scaling:** Numerical variables are standardized to improve model performance.

**Critical Code:**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
data['Category'] = encoder.fit_transform(data['Category'])
```
This snippet uses `LabelEncoder` to convert categorical variables into a numerical format suitable for machine learning models.

```python
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])
```
Here, `StandardScaler` standardizes numerical features, ensuring a mean of 0 and a standard deviation of 1.

### 2. Exploratory Data Analysis (EDA)

EDA reveals patterns, correlations, and distributions in the data. It includes:

- **Correlation Heatmaps**
- **Distribution Plots**
- **Churn Rate Analysis Across Features**

### 3. Model Building and Evaluation

The notebook constructs machine learning models to predict customer churn, using steps such as:

- **Data Splitting:** Separating data into training and test sets.
- **Model Training:** Implementing algorithms like Logistic Regression or Random Forest.
- **Evaluation Metrics:** Assessing models using accuracy, precision, recall, and F1-score.

**Critical Code:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```
This cell splits data and trains a Random Forest model on the training set.

### 4. Noise-Reduced Synthetic Minority Oversampling Technique (SMOTE)

To address the imbalanced classes, the notebook uses a noise-reduced version of SMOTE. This approach balances the dataset by:

- Identifying and removing noisy or outlier points using the Local Outlier Factor (LOF).
- Generating synthetic samples for the minority class.

**Critical Code:**

``` python
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor()
outliers = lof.fit_predict(X)
X_filtered, y_filtered = X[outliers == 1], y[outliers == 1]
```
This code identifies and excludes outliers before applying SMOTE.

```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)
```
After outlier removal, SMOTE creates synthetic samples to balance the data.

## Requirements

The notebook requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

