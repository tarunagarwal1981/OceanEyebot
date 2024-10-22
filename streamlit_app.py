import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import urllib.parse
import folium
from streamlit_folium import st_folium
import searoute as sr
from fuzzywuzzy import process
from datetime import date
import openai

from config import (
    DB_CONFIG, EMISSION_FACTORS, VESSEL_TYPE_MAPPING,
    CII_PARAMETERS, CII_REDUCTION_FACTORS, PAGE_CONFIG, CUSTOM_CSS
)
from chat_utils import (
    get_api_key, create_context, get_llm_response,
    initialize_chat_history, save_chat_history, analyze_query,
    format_response
)

# Initialize OpenAI
openai.api_key = get_api_key()

# Page configuration
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'cii_data' not in st.session_state:
    st.session_state.cii_data = {}
if 'port_table_data' not in st.session_state:
    st.session_state.port_table_data = []
if 'voyage_calculations' not in st.session_state:
    st.session_state.voyage_calculations = []

def get_db_engine():
    """Create and return database engine"""
    encoded_password = urllib.parse.quote(DB_CONFIG['password'])
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(db_url)

def calculate_reference_cii(capacity, ship_type):
    """Calculate reference CII based on capacity and ship type"""
    params = CII_PARAMETERS.get(ship_type.lower())
    if not params:
        raise ValueError(f"Unknown ship type: {ship_type}")
    
    for param in params:
        if capacity <= param['capacity_threshold']:
            return param['a'] * (capacity ** -param['c'])
    return params[-1]['a'] * (capacity ** -params[-1]['c'])

def calculate_required_cii(reference_cii, year):
    """Calculate required CII based on reference CII and year"""
    return reference_cii * CII_REDUCTION_FACTORS.get(year, 1.0)

def calculate_cii_rating(attained_cii, required_cii):
    """Calculate CII rating based on attained and required CII"""
    if attained_cii <= required_cii:
        return 'A'
    elif attained_cii <= 1.05 * required_cii:
        return 'B'
    elif attained_cii <= 1.1 * required_cii:
        return 'C'
    elif attained_cii <= 1.15 * required_cii:
        return 'D'
    else:
        return 'E'

@st.cache_data
def load_world_ports():
    """Load and cache world ports data"""
    return pd.read_csv("UpdatedPub150.csv")

def calculate_segment_metrics(row, world_ports_data):
    """Calculate metrics for a single voyage segment"""
    if not all([row[0], row[1], row[2], row[3], row[4], row[5]]):
        return None
    
    try:
        distance = route_distance(row[0], row[1], world_ports_data)
        sea_time = distance / (row[3] * 24)
        total_time = sea_time + row[2]
        co2_emissions = row[4] * sea_time * EMISSION_FACTORS[row[5]]
        
        return {
            'from_port': row[0],
            'to_port': row[1],
            'distance': distance,
            'sea_time': sea_time,
            'port_time': row[2],
            'total_time': total_time,
            'speed': row[3],
            'fuel_used': row[4],
            'fuel_type': row[5],
            'co2_emissions': co2_emissions
        }
    except Exception as e:
        st.error(f"Error calculating segment metrics: {str(e)}")
        return None

def run_calculator():
    """Run the CII calculator interface"""
    st.title('🚢 CII Calculator')
    
    # Load world ports data
    world_ports_data = load_world_ports()

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        vessel_name = st.text_input("Enter Vessel Name")
    with col2:
        year = st.number_input('Year for CII Calculation', 
                              min_value=2023, 
                              max_value=date.today().year, 
                              value=date.today().year)
    with col3:
        calculate_clicked = st.button('Calculate Current CII')

    if calculate_clicked and vessel_name:
        engine = get_db_engine()
        df = get_vessel_data(engine, vessel_name, year)
        process_vessel_data(df)

    # Display current CII results if available
    if st.session_state.cii_data:
        display_cii_results()

    # Voyage Planning Section
    st.markdown("### Voyage Planning")
    left_col, right_col = st.columns([6, 4])
    
    with left_col:
        handle_route_planning(world_ports_data)
    
    with right_col:
        display_route_map(world_ports_data)

def run_chat_interface():
    """Run the chat interface"""
    st.markdown("### CII Analysis Assistant")
    
    if not st.session_state.get('cii_data'):
        st.info("Please calculate CII data first in the Calculator tab.")
        return
    
    initialize_chat_history()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your CII analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Determine analysis type and create context
        analysis_type = analyze_query(prompt)
        context = create_context(
            st.session_state.cii_data,
            st.session_state.get('voyage_calculations', None)
        )

        # Get and display response
        with st.chat_message("assistant"):
            response = get_llm_response(prompt, context, analysis_type)
            formatted_response = format_response(response)
            st.markdown(formatted_response)
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})

    # Export chat button
    if len(st.session_state.messages) > 1:
        if st.button("Export Chat History"):
            chat_history = save_chat_history()
            if chat_history:
                st.download_button(
                    "Download Chat History",
                    chat_history,
                    file_name=f"cii_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

def main():
    """Main application entry point"""
    tab1, tab2 = st.tabs(["CII Calculator", "Chat Assistant"])
    
    with tab1:
        run_calculator()
    
    with tab2:
        run_chat_interface()

if __name__ == "__main__":
    main()
