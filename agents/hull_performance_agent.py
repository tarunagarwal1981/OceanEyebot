import pandas as pd
from utils.database_utils import fetch_data_from_db
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def plot_hull_roughness(vessel_name, data):
    if data.empty:
        return None
    
    # Convert report_date to datetime
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    # Filter data for the last 6 months and non-null power loss values
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        return None
    
    # Extract dates and power loss for plotting
    dates = pd.to_datetime(filtered_data['report_date'])
    power_loss = filtered_data['hull_roughness_power_loss']
    
    # Convert dates to numeric values for fitting
    x_numeric = (dates - dates.min()).dt.days
    
    # Create scatter plot and best fit line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)
    
    # Fit a best-fit line
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    # Smooth the line
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')
    
    # Style adjustments
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    
    # Set axis labels and title
    ax.set_xlabel('Dates', fontsize=12, color='white')
    ax.set_ylabel('Excess Power %', fontsize=12, color='white')
    ax.set_title(f'Hull Roughness Power Loss - {vessel_name}', fontsize=14, color='white')
    
    # Format date ticks
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    # Adjust ticks appearance
    plt.xticks(color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    
    # Set limits for x and y axes
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(power_loss.min() - 0.05 * (power_loss.max() - power_loss.min()), power_loss.max() + 0.05 * (power_loss.max() - power_loss.min()))
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    return fig

def analyze_hull_performance(vessel_name: str):
    # SQL query to fetch hull performance data for the vessel
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    
    # Fetch data from the database
    data = fetch_data_from_db(query)
    
    # Check if data was fetched successfully
    if data.empty:
        return f"No hull performance data available for {vessel_name}.", None, None, None
    
    # Call the plot_hull_roughness function
    fig = plot_hull_roughness(vessel_name, data)
    
    if fig is None:
        return f"No valid hull roughness data available for {vessel_name}.", None, None, None
    
    # Compute power loss statistics
    avg_power_loss = data['hull_roughness_power_loss'].mean()
    hull_condition = "Good" if avg_power_loss < 15 else ("Average" if avg_power_loss < 25 else "Poor")
    
    # Return analysis summary, average power loss, hull condition, and the figure
    return f"Hull performance for {vessel_name}: Average power loss is {avg_power_loss:.2f}%. Hull condition is {hull_condition}.", avg_power_loss, hull_condition, fig
