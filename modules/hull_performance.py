import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.database_utils import fetch_data_from_db

def fetch_performance_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def fetch_six_months_data(vessel_name):
    query = f"""
    SELECT vessel_name, hull_rough_power_loss_pct_ed
    FROM hull_performance_six_months
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def plot_hull_roughness(vessel_name, data):
    if data.empty:
        return None
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        return None
    
    dates = pd.to_datetime(filtered_data['report_date'])
    power_loss = filtered_data['hull_roughness_power_loss']
    
    x_numeric = (dates - dates.min()).dt.days
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)
    
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')
    
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    
    ax.set_xlabel('Dates', fontsize=12, color='white')
    ax.set_ylabel('Excess Power %', fontsize=12, color='white')
    ax.set_title(f'Hull Roughness Power Loss - {vessel_name}', fontsize=14, color='white')
    
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    plt.xticks(color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(power_loss.min() - 0.05 * (power_loss.max() - power_loss.min()), power_loss.max() + 0.05 * (power_loss.max() - power_loss.min()))
    
    ax.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    return fig

def get_hull_condition(power_loss_pct):
    if power_loss_pct > 25:
        return "Poor"
    elif 15 <= power_loss_pct <= 25:
        return "Average"
    else:
        return "Good"

def analyze_hull_performance(vessel_name):
    performance_data = fetch_performance_data(vessel_name)
    six_months_data = fetch_six_months_data(vessel_name)
    
    chart = plot_hull_roughness(vessel_name, performance_data)
    
    if not six_months_data.empty:
        power_loss_pct_ed = six_months_data['hull_rough_power_loss_pct_ed'].iloc[-1]
        hull_condition = get_hull_condition(power_loss_pct_ed)
    else:
        power_loss_pct_ed = None
        hull_condition = None
    
    return chart, power_loss_pct_ed, hull_condition
