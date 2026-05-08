# FIRST

# Titanic Survival Prediction Project

This project is a Machine Learning based Titanic Survival Prediction system built using Python and Scikit-learn. The goal of the project is to predict whether a passenger survived the Titanic disaster based on different passenger features.

## What I Did in This Project

### 1. Data Loading

* Loaded the Titanic dataset using Pandas.
* Used train.csv dataset from Kaggle.

### 2. Data Preprocessing

* Checked null/missing values in the dataset.
* Handled missing values using:

  * Mean imputation for Age
  * Most frequent value for Embarked
  * Filled Cabin column with "Unknown"

### 3. Data Cleaning

* Removed unnecessary columns:

  * Cabin
  * Name
  * Ticket
* These columns were removed because they contained many missing values or were less useful for prediction.

### 4. Feature Encoding

* Converted categorical values into numerical format:

  * male → 0
  * female → 1
* Applied One-Hot Encoding on the Embarked column.

### 5. Exploratory Data Analysis (EDA)

Performed visualization and analysis using:

* Plotly
* Seaborn
* Matplotlib

Visualizations created:

* Distribution of survived passengers
* Distribution of passenger class
* Passenger class vs survival
* Embarked location vs survival
* Correlation analysis between features

### 6. Feature Scaling

* Applied StandardScaler to normalize the feature values before model training.

### 7. Train-Test Split

* Split the dataset into:

  * Training data
  * Testing data
* Used train_test_split from Scikit-learn.

### 8. Model Training

* Trained the model using Logistic Regression.

### 9. Model Testing and Evaluation

Evaluated model performance using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 10. Custom Prediction Testing

* Tested the model on custom passenger data to predict survival.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Seaborn
* Matplotlib

## Outcome

Successfully built an end-to-end Machine Learning project including:

* Data preprocessing
* EDA
* Feature engineering
* Model training
* Model testing
* Prediction system

This project helped me understand the complete Machine Learning workflow from raw dataset to prediction and evaluation.

