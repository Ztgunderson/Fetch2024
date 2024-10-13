from flask import Flask, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all domains (adjust as needed)

# Ensure the static/charts directory exists
CHARTS_DIR = os.path.join(app.root_path, 'static', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Load your pre-trained model
# Ensure that 'model.joblib' is in the backend directory
model = joblib.load('model.joblib')  # Assuming it's a simple linear model: [intercept, slope]

def load_and_process_data():
    # Load historical data
    df = pd.read_csv('data_daily.csv')
    
    # Convert '# Date' column to datetime and set as index
    df['# Date'] = pd.to_datetime(df['# Date'])
    df.set_index('# Date', inplace=True)
    
    # Prepare features and target
    dates_ordinal = np.array([d.toordinal() for d in df.index])
    y = df['Receipt_Count'].to_numpy()
    X = dates_ordinal.reshape(-1, 1)
    
    return df, X, y

def generate_predictions():
    predictions = []
    monthly_sums = {}
    
    # Start date: January 1, 2022
    start_date = datetime(2022, 1, 1)
    
    for i in range(365):
        date = start_date + timedelta(days=i)
        predicted_value = model[0] + model[1] * date.toordinal()
        predictions.append({'date': date.strftime('%Y-%m-%d'), 'value': predicted_value})
        
        year = date.year
        month = date.month
        
        if (year, month) not in monthly_sums:
            monthly_sums[(year, month)] = 0
        monthly_sums[(year, month)] += predicted_value
    
    # Format monthly totals
    monthly_totals = [
        {'year': year, 'month': calendar.month_name[month], 'total': total}
        for (year, month), total in monthly_sums.items()
    ]
    
    return predictions, monthly_totals

def create_line_chart(df, predictions):
    plt.figure(figsize=(14, 7))
    
    # Plot historical data (2021)
    plt.plot(df.index, df['Receipt_Count'], label='Historical Receipts (2021)', color='blue')
    
    # Plot predictions (2022)
    pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions]
    pred_values = [p['value'] for p in predictions]
    plt.plot(pred_dates, pred_values, label='Predicted Receipts (2022)', color='orange')
    
    plt.title('Daily Receipts: 2021 vs. 2022 Forecast')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as an image
    line_chart_path = os.path.join(CHARTS_DIR, 'daily_receipts.png')
    plt.savefig(line_chart_path)
    plt.close()
    
    return '/static/charts/daily_receipts.png'

def create_bar_chart(monthly_totals):
    plt.figure(figsize=(14, 7))
    
    # Extract months and totals
    months = [f"{mt['month']} {mt['year']}" for mt in monthly_totals]
    totals = [mt['total'] for mt in monthly_totals]
    
    # Plot bar chart
    plt.bar(months, totals, color='green')
    plt.title('Monthly Total Forecasted Receipts (2022)')
    plt.xlabel('Month')
    plt.ylabel('Total Receipts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as an image
    bar_chart_path = os.path.join(CHARTS_DIR, 'monthly_totals.png')
    plt.savefig(bar_chart_path)
    plt.close()
    
    return '/static/charts/monthly_totals.png'

# Load and process data
df, X, y = load_and_process_data()

# Generate predictions
yearly_predictions, monthly_totals = generate_predictions()

# Create charts
line_chart_url = create_line_chart(df, yearly_predictions)
bar_chart_url = create_bar_chart(monthly_totals)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')