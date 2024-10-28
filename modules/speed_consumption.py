# speed_consumption.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.database_utils import fetch_data_from_db

# Fetch the speed consumption data from the database
def fetch_speed_consumption_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

# Plot speed vs. normalised consumption for Laden and Ballast conditions
def plot_speed_consumption(vessel_name, data):
    if data.empty:
        return None
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    # Filter the data for the last 6 months
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago)]
    
    if filtered_data.empty:
        return None
    
    # Separate data by loading condition (laden and ballast)
    laden_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'laden']
    ballast_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'ballast']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Laden Condition
    if not laden_data.empty:
        laden_dates = pd.to_datetime(laden_data['report_date'])
        x_laden = laden_data['speed']
        y_laden = laden_data['normalised_consumption']
        scatter_laden = ax1.scatter(x_laden, y_laden, c=(laden_dates - laden_dates.min()).dt.days, cmap='plasma', s=50, alpha=0.8)
        ax1.set_title('Laden Condition')
        ax1.set_xlabel('Speed (knots)')
        ax1.set_ylabel('ME Consumption (mT/d)')
        plt.colorbar(scatter_laden, ax=ax1, label="Time Progression")
    
    # Ballast Condition
    if not ballast_data.empty:
        ballast_dates = pd.to_datetime(ballast_data['report_date'])
        x_ballast = ballast_data['speed']
        y_ballast = ballast_data['normalised_consumption']
        scatter_ballast = ax2.scatter(x_ballast, y_ballast, c=(ballast_dates - ballast_dates.min()).dt.days, cmap='plasma', s=50, alpha=0.8)
        ax2.set_title('Ballast Condition')
        ax2.set_xlabel('Speed (knots)')
        plt.colorbar(scatter_ballast, ax=ax2, label="Time Progression")
    
    plt.tight_layout()
    return fig

# Analyze speed consumption for the vessel
def analyze_speed_consumption(vessel_name):
    # Fetch data
    speed_data = fetch_speed_consumption_data(vessel_name)
    
    # Plot speed consumption
    chart = plot_speed_consumption(vessel_name, speed_data)
    
    return chart
