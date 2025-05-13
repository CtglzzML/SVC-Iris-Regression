# SVM Diabetes Regression

This project demonstrates the use of Support Vector Regression (SVR) on the Diabetes dataset included in scikit-learn. The objective is to predict the progression of diabetes based on 10 baseline variables.

## Dataset

The Diabetes dataset is a regression dataset with 442 samples and 10 features. The target variable is a quantitative measure of disease progression one year after baseline.

## Objective

To build and evaluate SVR models using different kernel types (`linear`, `poly`, `rbf`, and `sigmoid`), and compare their performance based on the root mean squared error (RMSE).

## Technologies

- Python
- scikit-learn
- pandas
- numpy

## Output

The script prints the RMSE for each SVR kernel on the test set after standardizing the input features.

