# Customer Churn Prediction
## Project Overview
Model (model.ipynb):
- This file handles the machine learning model: loading the models, preprocessing the data, and predicting churn probabilities.

Application (app.py):
- This file contains the Dash web application.

This project involves building a customer churn prediction system using machine learning and developing an interactive Dash application to visualize the predictions and insights. The goal of this system is to predict whether a customer is likely to churn (leave the service) and take actions accordingly. The app provides insights into the churn predictions, including a table of the top 10 most probable churn customers and a pie chart for the overall churn distribution.

## Why This Project is relevant for MadKudu?

The dataset used in this project is the Telco Customer Churn Dataset. This dataset is highly relevant for MadKudu’s business, as it contains real customer data from a telecommunications company, including various features like customer demographics, account information, and service usage. MadKudu specializes in customer lifecycle management and predictive analytics. By predicting churn, MadKudu can help companies target high-risk customers with personalized offers, retention campaigns, or interventions to reduce churn.

### This dataset includes:
- Customer information: age, gender, contract type, etc.
- Service usage information: tenure, monthly charges, payment method, etc.
- Churn label: Whether the customer has left the service (1 for churn, 0 for non-churn).

## Model Selection

I experimented with several models, including logistic regression and random forests, but they didn't perform well in identifying churners. XGBoost emerged as the best model, despite having lower precision for churn predictions, because it achieved high recall.
- Recall Priority: In churn prediction, recall is more important than precision because the goal is to identify as many churners as possible, even if it means some false positives. XGBoost excelled in this area, capturing most churners.
- Class Imbalance: XGBoost handled the class imbalance well using the scale_pos_weight parameter, which helped focus more on the minority class (churners).

While precision was lower, XGBoost's strong recall made it the most effective choice for predicting churn and taking proactive retention actions.


## How to run the App

```
python app.py
```
The app has two dashboards:
- Top 10 Most Probable Churn Customers: Displays the top 10 customers most likely to churn.
- Churn Prediction Distribution: Shows a pie chart of churn predictions across all customers.