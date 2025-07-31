# Telecom Customer Churn Prediction

This is a machine learning project where I worked on predicting customer churn for a telecom company. The goal is to identify whether a customer is likely to leave the service based on their usage patterns and account information.

## About the Project

Customer churn is a major issue for telecom companies. If we can predict which customers are at risk of leaving, the company can take steps to retain them. This project uses real-world data to build a predictive model that helps solve that problem.

## Dataset

The dataset was taken from Kaggle and contains information like customer demographics, services signed up for, payment methods, and churn status.

- Total records: 7043
- Target column: `Churn` (Yes or No)

## Steps I Followed

1. **Data Cleaning**: Handled missing values and converted categorical columns to numerical values.
2. **Exploratory Data Analysis (EDA)**: Understood which features influence churn the most.
3. **Model Training**: Tried Logistic Regression and Decision Tree classifiers.
4. **Evaluation**: Measured accuracy, confusion matrix, and classification report.
5. **Conclusion**: Highlighted which features contribute the most to churn.

## Libraries Used

- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## How to Run

1. Download the dataset from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place the CSV file in the same folder as the Python script.
3. Run the script using any Python IDE or Jupyter Notebook.

python Telecom_churn_prediction.py
