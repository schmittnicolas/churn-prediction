import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os
import random

model = xgb.Booster()
model.load_model('xgboost_model.json')

dataset_path = 'churn_predictions.csv'

df = pd.read_csv(dataset_path)
df['CustomerID'] = range(1, len(df) + 1)
df["ContactInfo"] = df["CustomerID"].apply(lambda x: f"{x}@example.com")


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Customer Churn Prediction Dashboard"),
    
    dcc.Input(id='dummy-input', style={'display': 'none'}, value='trigger'),
    
    html.H3("Top 10 Most Probable Churn Customers"),
    html.Div(id='top-10-table'),
    
    html.H3("Churn Prediction Distribution"),
    dcc.Graph(id='churn-distribution')
])


@app.callback(
    [Output('top-10-table', 'children'),
     Output('churn-distribution', 'figure')],
    [Input('dummy-input', 'value')]
)
def update_dashboard(dummy_value):
    top_10_df = df.nlargest(10, 'ChurnProbability')
    col_to_display = ['CustomerID', 'ChurnProbability', 'ChurnPrediction', 'ContactInfo']
    table = html.Table([
        html.Tr([html.Th(col) for col in col_to_display])
    ] + [
        html.Tr([html.Td(top_10_df.iloc[i][col]) for col in col_to_display]) for i in range(min(len(top_10_df), 10))
    ])
    
    churn_counts = df['ChurnPrediction'].value_counts()
    churn_fig = px.pie(names=churn_counts.index, values=churn_counts.values, title="Churn Prediction Distribution")
    
    return table, churn_fig

if __name__ == '__main__':
    app.run_server(debug=True)