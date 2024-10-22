import os
from datetime import date

# Database configuration
DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

# Emission factors for different fuel types
EMISSION_FACTORS = {
    'VLSFO': 3.151,
    'LSMGO': 3.206,
    'LNG': 2.75
}

# Vessel type mapping
VESSEL_TYPE_MAPPING = {
    'ASPHALT/BITUMEN TANKER': 'tanker',
    'BULK CARRIER': 'bulk_carrier',
    'CEMENT CARRIER': 'bulk_carrier',
    'CHEM/PROD TANKER': 'tanker',
    'CHEMICAL TANKER': 'tanker',
    'Chemical/Products Tanker': 'tanker',
    'Combination Carrier': 'combination_carrier',
    'CONTAINER': 'container_ship',
    'Container Ship': 'container_ship',
    'Container/Ro-Ro Ship': 'ro_ro_cargo_ship',
    'Crude Oil Tanker': 'tanker',
    'Gas Carrier': 'gas_carrier',
    'General Cargo Ship': 'general_cargo_ship',
    'LNG CARRIER': 'lng_carrier',
    'LPG CARRIER': 'gas_carrier',
    'LPG Tanker': 'gas_carrier',
    'OIL TANKER': 'tanker',
    'Products Tanker': 'tanker',
    'Refrigerated Cargo Ship': 'refrigerated_cargo_carrier',
    'Ro-Ro Ship': 'ro_ro_cargo_ship',
    'Vehicle Carrier': 'ro_ro_cargo_ship_vc'
}

# CII Reference Parameters
CII_PARAMETERS = {
    'bulk_carrier': [{'capacity_threshold': 279000, 'a': 4745, 'c': 0.622}],
    'gas_carrier': [{'capacity_threshold': 65000, 'a': 144050000000, 'c': 2.071}],
    'tanker': [{'capacity_threshold': float('inf'), 'a': 5247, 'c': 0.61}],
    'container_ship': [{'capacity_threshold': float('inf'), 'a': 1984, 'c': 0.489}],
    'general_cargo_ship': [{'capacity_threshold': float('inf'), 'a': 31948, 'c': 0.792}],
    'refrigerated_cargo_carrier': [{'capacity_threshold': float('inf'), 'a': 4600, 'c': 0.557}],
    'lng_carrier': [{'capacity_threshold': 100000, 'a': 144790000000000, 'c': 2.673}],
}

# CII Reduction Factors
CII_REDUCTION_FACTORS = {
    2023: 0.95,
    2024: 0.93,
    2025: 0.91,
    2026: 0.89
}

# Chat System Messages
SYSTEM_MESSAGES = {
    'general': """You are a maritime emissions expert assistant specialized in CII (Carbon Intensity Indicator) analysis. 
    Your role is to help users understand their vessel's carbon intensity performance and provide actionable insights.
    
    When responding:
    1. Reference specific data points from the provided context
    2. Explain technical terms in simple language
    3. Provide practical recommendations
    4. Consider IMO regulations and industry standards
    5. Be concise but thorough in your explanations""",
    
    'analysis': """Analyze the provided CII data and explain:
    1. Current rating and its implications for compliance
    2. Comparison with required CII values
    3. Key factors affecting the rating
    4. Areas for potential improvement
    5. Regulatory requirements and deadlines
    
    Use actual numbers from the context and provide specific recommendations.""",
    
    'voyage': """Review the voyage plan and provide:
    1. Projected impact on annual CII rating
    2. Route efficiency analysis
    3. Fuel consumption patterns and optimization opportunities
    4. Specific recommendations for:
       - Speed optimization
       - Port time management
       - Fuel type selection
       - Route planning
    
    Base your analysis on the provided voyage calculations and current CII status."""
}

# Streamlit page config
PAGE_CONFIG = {
    'page_title': "CII Calculator & Analyzer",
    'page_icon': "🚢",
    'layout': "wide"
}

# Custom CSS
CUSTOM_CSS = """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
"""
