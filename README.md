# House Prices: Advanced Regression Techniques - Kaggle Competition

## Overview
This repository contains a solution for the Kaggle competition "House Prices: Advanced Regression Techniques" (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal of this competition is to predict house sale prices based on a dataset containing various features about houses in Ames, Iowa. This solution employs advanced feature engineering, feature selection, and machine learning models to achieve accurate predictions.

## Approach
The solution follows a structured machine learning pipeline:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
2. **Feature Engineering**: Creating new features such as total square footage, house age, remodel age, and total bathrooms to capture additional information.
3. **Feature Selection**: Using Random Forest to identify the most important features based on importance thresholds.
4. **Model Selection**: Evaluating multiple regression models and tuning the best-performing model (Gradient Boosting Regressor).
5. **Prediction**: Generating predictions on the test set and preparing a submission file.

## Requirements
To run the code, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `scipy`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm scipy
```

## Dataset
The dataset is provided by Kaggle and includes:
- `train.csv`: Training data with features and target variable (`SalePrice`).
- `test.csv`: Test data for generating predictions.

Place these files in a `data/` directory within the project folder.

## Code Structure
The solution is implemented in a Jupyter Notebook (`house_prices.ipynb`) with the following key sections:
1. **Initialization**: Importing libraries and loading the dataset.
2. **Data Preprocessing**: Handling missing values and splitting data into train/dev sets.
3. **Feature Engineering**:
   - Numerical features: Total square footage, house age, remodel age, total bathrooms.
   - Ordinal features: Mapping categorical variables with inherent order (e.g., quality ratings).
   - Nominal features: Label encoding for categorical variables without order.
   - Skewness correction: Log-transforming skewed numerical features.
4. **Feature Selection**: Using Random Forest to select the top features (59 features selected with a threshold of 0.7).
5. **Model Selection**: Comparing models including Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM. Gradient Boosting (GBM) was selected as the best model.
6. **Hyperparameter Tuning**: Using RandomizedSearchCV to optimize GBM parameters.
7. **Final Prediction**: Training the tuned model on the full training set and predicting on the test set.

## Results
- **Best Model**: Gradient Boosting Regressor (GBM)
- **Best Parameters**: 
  ```python
  {
      'subsample': 0.8,
      'n_estimators': 100,
      'min_samples_split': 5,
      'min_samples_leaf': 4,
      'max_features': 'sqrt',
      'max_depth': 3,
      'loss': 'huber',
      'learning_rate': 0.1
  }
  ```
- **Performance**:
  - RMSE (log scale) on dev set: 0.1322
  - RMSE (original scale) on dev set: ~25,979 USD

The final predictions are saved in `submission_gbm.csv`.

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Place `train.csv` and `test.csv` in the `data/` directory.
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook house_prices.ipynb
   ```
4. Run all cells to preprocess the data, train the model, and generate the submission file.

## Submission
The submission file (`submission_gbm.csv`) contains two columns:
- `Id`: House ID from the test set.
- `SalePrice`: Predicted sale price.

Upload this file to the Kaggle competition page to evaluate the performance on the test set.

## Future Improvements
- Experiment with additional feature engineering (e.g., interaction terms, polynomial features).
- Explore ensemble methods combining multiple models (e.g., stacking).
- Fine-tune hyperparameters further with a larger grid or Bayesian optimization.

## Acknowledgments
- Kaggle for providing the dataset and competition platform.
- The open-source community for the libraries used (scikit-learn, XGBoost, LightGBM).


