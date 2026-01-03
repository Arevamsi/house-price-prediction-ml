🏡 Melbourne House Price Prediction using Random Forest

This project demonstrates how to build a machine learning regression model to predict house prices in Melbourne using the Random Forest algorithm. The dataset is processed, trained, evaluated, and visualized step by step using Python.

📌 Project Overview

Goal: Predict house prices based on property features

Algorithm Used: Random Forest Regressor

Evaluation Metric: Mean Absolute Error (MAE)

Visualization: Actual vs Predicted price comparison

📂 Dataset

File: melb_data.csv

Source: Melbourne housing dataset

Target Variable: Price

Features Used:

Rooms

Bathroom

Landsize

BuildingArea

YearBuilt

Lattitude

Longtitude

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🔄 Workflow
1. Load the Data

Load the CSV file using Pandas

Inspect the dataset structure

2. Data Preparation

Remove rows with missing target values (Price)

Select numerical features

Handle missing values using mean imputation

Split data into training and validation sets

3. Model Training

Train a RandomForestRegressor on the training data

4. Prediction

Generate predictions on the validation dataset

5. Model Evaluation

Evaluate performance using Mean Absolute Error (MAE)

Achieved MAE: ~175,216

6. Visualization

Scatter plot of Actual vs Predicted Prices

Diagonal line shows perfect predictions

📊 Results

The model predicts house prices with reasonable accuracy

Visualization helps identify prediction errors and trends

Random Forest performs well on structured housing data

📁 Project Structure
melbourne-house-price-prediction/
│
├── melb_data.csv
├── house_price_prediction.ipynb
├── README.md

🚀 How to Run the Project

Clone the repository

Install required libraries:

pip install pandas scikit-learn matplotlib seaborn


Open and run the notebook:

jupyter notebook

📌 Future Improvements

Add categorical feature encoding

Try advanced models (XGBoost, Gradient Boosting)

Perform hyperparameter tuning

Add cross-validation
