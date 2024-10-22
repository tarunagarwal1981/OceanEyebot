import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import urllib.parse
import folium
from streamlit_folium import st_folium
import searoute as sr
from fuzzywuzzy import process
from datetime import date, datetime
import openai
import os
import json

# Database configuration
DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

# Emission factors
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

# CII Parameters
CII_PARAMETERS = {
    'bulk_carrier': [{'capacity_threshold': 279000, 'a': 4745, 'c': 0.622}],
    'gas_carrier': [{'capacity_threshold': 65000, 'a': 144050000000, 'c': 2.071}],
    'tanker': [{'capacity_threshold': float('inf'), 'a': 5247, 'c': 0.61}],
    'container_ship': [{'capacity_threshold': float('inf'), 'a': 1984, 'c': 0.489}],
    'general_cargo_ship': [{'capacity_threshold': float('inf'), 'a': 31948, 'c': 0.792}],
    'refrigerated_cargo_carrier': [{'capacity_threshold': float('inf'), 'a': 4600, 'c': 0.557}],
    'lng_carrier': [{'capacity_threshold': 100000, 'a': 144790000000000, 'c': 2.673}]
}

# System messages for chat
SYSTEM_MESSAGES = {
    'general': """You are a maritime emissions expert assistant. You help analyze CII (Carbon Intensity Indicator) data 
    and provide insights. Use the actual numbers from the context in your explanation.""",
    'analysis': """Analyze the provided CII data and explain the current rating, comparison with required CII, 
    key factors affecting the rating, and potential improvement areas.""",
    'voyage': """Review the voyage plan and provide impact on annual CII rating, route efficiency analysis, 
    and specific recommendations for improvement."""
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
    </style>
"""

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'cii_data' not in st.session_state:
    st.session_state.cii_data = {}
if 'port_table_data' not in st.session_state:
    st.session_state.port_table_data = []
if 'voyage_calculations' not in st.session_state:
    st.session_state.voyage_calculations = []

def get_api_key():
    """Retrieve OpenAI API key"""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

def get_db_engine():
    """Create and return database engine"""
    encoded_password = urllib.parse.quote(DB_CONFIG['password'])
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(db_url)

def get_vessel_data(engine, vessel_name, year):
    """Fetch vessel data from database"""
    query = text("""
    SELECT 
        t1."VESSEL_NAME" AS "Vessel",
        t1."VESSEL_IMO" AS "IMO",
        SUM("DISTANCE_TRAVELLED_ACTUAL") AS "total_distance",
        COALESCE((SUM("FUEL_CONSUMPTION_HFO") - SUM("FC_FUEL_CONSUMPTION_HFO")) * 3.114, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_LFO") - SUM("FC_FUEL_CONSUMPTION_LFO")) * 3.151, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_GO_DO") - SUM("FC_FUEL_CONSUMPTION_GO_DO")) * 3.206, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_LNG") - SUM("FC_FUEL_CONSUMPTION_LNG")) * 2.75, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_LPG") - SUM("FC_FUEL_CONSUMPTION_LPG")) * 3.00, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_METHANOL") - SUM("FC_FUEL_CONSUMPTION_METHANOL")) * 1.375, 0) + 
        COALESCE((SUM("FUEL_CONSUMPTION_ETHANOL") - SUM("FC_FUEL_CONSUMPTION_ETHANOL")) * 1.913, 0) AS "CO2Emission",
        t2."deadweight" AS "capacity",
        t2."vessel_type",
        ROUND(CAST(SUM("DISTANCE_TRAVELLED_ACTUAL") * t2."deadweight" AS NUMERIC), 2) AS "Transportwork",
        CASE 
            WHEN ROUND(CAST(SUM("DISTANCE_TRAVELLED_ACTUAL") * t2."deadweight" AS NUMERIC), 2) <> 0 
            THEN ROUND(CAST((
                COALESCE((SUM("FUEL_CONSUMPTION_HFO") - SUM("FC_FUEL_CONSUMPTION_HFO")) * 3.114, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_LFO") - SUM("FC_FUEL_CONSUMPTION_LFO")) * 3.151, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_GO_DO") - SUM("FC_FUEL_CONSUMPTION_GO_DO")) * 3.206, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_LNG") - SUM("FC_FUEL_CONSUMPTION_LNG")) * 2.75, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_LPG") - SUM("FC_FUEL_CONSUMPTION_LPG")) * 3.00, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_METHANOL") - SUM("FC_FUEL_CONSUMPTION_METHANOL")) * 1.375, 0) + 
                COALESCE((SUM("FUEL_CONSUMPTION_ETHANOL") - SUM("FC_FUEL_CONSUMPTION_ETHANOL")) * 1.913, 0)
            ) * 1000000 / (SUM("DISTANCE_TRAVELLED_ACTUAL") * t2."deadweight") AS NUMERIC), 2)
            ELSE NULL
        END AS "Attained_AER"
    FROM 
        "sf_consumption_logs" AS t1
    LEFT JOIN 
        "vessel_particulars" AS t2 ON t1."VESSEL_IMO" = t2."vessel_imo"
    WHERE 
        t1."VESSEL_NAME" = :vessel_name
        AND EXTRACT(YEAR FROM "REPORT_DATE") = :year
    GROUP BY 
        t1."VESSEL_NAME", t1."VESSEL_IMO", t2."deadweight", t2."vessel_type"
    """)
    
    try:
        return pd.read_sql(query, engine, params={'vessel_name': vessel_name, 'year': year})
    except Exception as e:
        st.error(f"Error executing SQL query: {str(e)}")
        return pd.DataFrame()

def calculate_reference_cii(capacity, ship_type):
    """Calculate reference CII based on capacity and ship type"""
    params = CII_PARAMETERS.get(ship_type.lower())
    if not params:
        raise ValueError(f"Unknown ship type: {ship_type}")
    
    a, c = params[0]['a'], params[0]['c']
    return a * (capacity ** -c)

def calculate_required_cii(reference_cii, year):
    """Calculate required CII based on reference CII and year"""
    reduction_factors = {2023: 0.95, 2024: 0.93, 2025: 0.91, 2026: 0.89}
    return reference_cii * reduction_factors.get(year, 1.0)

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

def world_port_index(port_to_match, world_ports_data):
    """Find best matching port from world ports data"""
    best_match = process.extractOne(port_to_match, world_ports_data['Main Port Name'])
    return world_ports_data[world_ports_data['Main Port Name'] == best_match[0]].iloc[0]

def route_distance(origin, destination, world_ports_data):
    """Calculate route distance between two ports"""
    try:
        origin_port = world_port_index(origin, world_ports_data)
        destination_port = world_port_index(destination, world_ports_data)
        origin_coords = [float(origin_port['Longitude']), float(origin_port['Latitude'])]
        destination_coords = [float(destination_port['Longitude']), float(destination_port['Latitude'])]
        sea_route = sr.searoute(origin_coords, destination_coords, units="naut")
        return int(sea_route['properties']['length'])
    except Exception as e:
        st.error(f"Error calculating distance between {origin} and {destination}: {str(e)}")
        return 0

def plot_route(ports, world_ports_data):
    """Plot route on a Folium map"""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    if len(ports) >= 2 and all(ports):
        coordinates = []
        for i in range(len(ports) - 1):
            try:
                start_port = world_port_index(ports[i], world_ports_data)
                end_port = world_port_index(ports[i+1], world_ports_data)
                start_coords = [float(start_port['Latitude']), float(start_port['Longitude'])]
                end_coords = [float(end_port['Latitude']), float(end_port['Longitude'])]
                
                # Add markers for ports
                folium.Marker(
                    start_coords,
                    popup=ports[i],
                    icon=folium.Icon(color='green' if i == 0 else 'blue')
                ).add_to(m)
                
                if i == len(ports) - 2:
                    folium.Marker(
                        end_coords,
                        popup=ports[i+1],
                        icon=folium.Icon(color='red')
                    ).add_to(m)
                
                # Draw route line
                route = sr.searoute(start_coords[::-1], end_coords[::-1])
                folium.PolyLine(
                    locations=[list(reversed(coord)) for coord in route['geometry']['coordinates']], 
                    color="red",
                    weight=2,
                    opacity=0.8
                ).add_to(m)
                
                coordinates.extend([start_coords, end_coords])
            except Exception as e:
                st.error(f"Error plotting route for {ports[i]} to {ports[i+1]}: {str(e)}")
        
        if coordinates:
            m.fit_bounds(coordinates)
    
    return m

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

def handle_route_planning(world_ports_data):
    """Handle route planning input and calculations"""
    st.markdown("#### Route Information")
    
    # Create DataFrame for route information
    port_data_df = pd.DataFrame(
        st.session_state.port_table_data,
        columns=["From Port", "To Port", "Port Days", "Speed (knots)", 
                "Fuel Used (mT)", "Fuel Type"]
    )
    
    # Data editor for route planning
    edited_df = st.data_editor(
        port_data_df,
        num_rows="dynamic",
        key="port_table_editor",
        column_config={
            "From Port": st.column_config.TextColumn(
                "From Port",
                help="Enter departure port name",
                required=True
            ),
            "To Port": st.column_config.TextColumn(
                "To Port",
                help="Enter arrival port name",
                required=True
            ),
            "Port Days": st.column_config.NumberColumn(
                "Port Days",
                help="Enter number of days in port",
                min_value=0,
                max_value=100,
                step=0.5,
                required=True
            ),
            "Speed (knots)": st.column_config.NumberColumn(
                "Speed (knots)",
                help="Enter vessel speed in knots",
                min_value=1,
                max_value=30,
                step=0.1,
                required=True
            ),
            "Fuel Used (mT)": st.column_config.NumberColumn(
                "Fuel Used (mT)",
                help="Enter total fuel consumption",
                min_value=0,
                step=0.1,
                required=True
            ),
            "Fuel Type": st.column_config.SelectboxColumn(
                "Fuel Type",
                help="Select fuel type",
                options=list(EMISSION_FACTORS.keys()),
                required=True
            )
        }
    )
    
    # Update session state with edited data
    st.session_state.port_table_data = edited_df.values.tolist()

def display_route_map(world_ports_data):
    """Display route map using Folium"""
    if len(st.session_state.port_table_data) >= 1:
        ports = [row[0] for row in st.session_state.port_table_data if row[0]]
        if st.session_state.port_table_data[-1][1]:  # Add last destination
            ports.append(st.session_state.port_table_data[-1][1])
        
        if len(ports) >= 2:
            m = plot_route(ports, world_ports_data)
        else:
            m = folium.Map(location=[0, 0], zoom_start=2)
    else:
        m = folium.Map(location=[0, 0], zoom_start=2)
    
    st_folium(m, width=None, height=400)

def create_context(cii_data, voyage_calculations=None):
    """Create context for LLM"""
    context = {
        'current_cii': {
            'vessel_name': cii_data.get('vessel_name', ''),
            'attained_aer': cii_data.get('attained_aer', 0),
            'required_cii': cii_data.get('required_cii', 0),
            'rating': cii_data.get('cii_rating', ''),
            'total_distance': cii_data.get('total_distance', 0),
            'co2_emission': cii_data.get('co2_emission', 0),
            'vessel_type': cii_data.get('vessel_type', ''),
            'capacity': cii_data.get('capacity', 0)
        }
    }
    
    if voyage_calculations:
        context['planned_voyage'] = {
            'segments': voyage_calculations,
            'total_distance': sum(seg.get('distance', 0) for seg in voyage_calculations),
            'total_co2': sum(seg.get('co2_emissions', 0) for seg in voyage_calculations)
        }
    
    return json.dumps(context, indent=2)

def get_llm_response(user_query, context, analysis_type='general'):
    """Generate LLM response"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGES[analysis_type]},
            {"role": "user", "content": f"""
Context:
{context}

User Question: {user_query}

Provide a clear, concise response based on the above context."""}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

def analyze_query(query):
    """Determine the type of analysis needed"""
    query_lower = query.lower()
    if any(word in query_lower for word in ['voyage', 'route', 'plan', 'distance']):
        return 'voyage'
    elif any(word in query_lower for word in ['analyze', 'analysis', 'explain', 'why', 'how']):
        return 'analysis'
    return 'general'

def display_cii_results():
    """Display CII calculation results"""
    st.markdown("### Current CII Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Attained AER', 
                 f"{st.session_state.cii_data['attained_aer']:.4f}")
    with col2:
        st.metric('Required CII', 
                 f"{st.session_state.cii_data['required_cii']:.4f}")
    with col3:
        st.metric('CII Rating', 
                 st.session_state.cii_data['cii_rating'])
    with col4:
        st.metric('Total Distance (NM)', 
                 f"{st.session_state.cii_data['total_distance']:,.0f}")
    with col5:
        st.metric('CO2 Emission (MT)', 
                 f"{st.session_state.cii_data['co2_emission']:,.1f}")

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
        
        if not df.empty:
            vessel_type = df['vessel_type'].iloc[0]
            imo_ship_type = VESSEL_TYPE_MAPPING.get(vessel_type)
            capacity = df['capacity'].iloc[0]
            attained_aer = df['Attained_AER'].iloc[0]

            if imo_ship_type and attained_aer is not None:
                reference_cii = calculate_reference_cii(capacity, imo_ship_type)
                required_cii = calculate_required_cii(reference_cii, year)
                cii_rating = calculate_cii_rating(attained_aer, required_cii)
                
                st.session_state.cii_data = {
                    'vessel_name': df['Vessel'].iloc[0],
                    'attained_aer': attained_aer,
                    'required_cii': required_cii,
                    'cii_rating': cii_rating,
                    'total_distance': df['total_distance'].iloc[0],
                    'co2_emission': df['CO2Emission'].iloc[0],
                    'capacity': capacity,
                    'vessel_type': vessel_type,
                    'imo_ship_type': imo_ship_type
                }
            else:
                if imo_ship_type is None:
                    st.error(f"The vessel type '{vessel_type}' is not supported for CII calculations.")
                if attained_aer is None:
                    st.error("Unable to calculate Attained AER. Please check the vessel's data.")
        else:
            st.error(f"No data found for vessel {vessel_name} in year {year}")

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
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Export chat button
    if len(st.session_state.messages) > 1:
        if st.button("Export Chat History"):
            chat_history = {
                'timestamp': datetime.now().isoformat(),
                'vessel_name': st.session_state.cii_data.get('vessel_name', 'Unknown'),
                'messages': st.session_state.messages
            }
            st.download_button(
                "Download Chat History",
                json.dumps(chat_history, indent=2),
                file_name=f"cii_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

def main():
    """Main application entry point"""
    # Page config
    st.set_page_config(page_title="CII Calculator", layout="wide", page_icon="🚢")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize OpenAI
    openai.api_key = get_api_key()
    
    # Create tabs
    tab1, tab2 = st.tabs(["CII Calculator", "Chat Assistant"])
    
    with tab1:
        run_calculator()
    
    with tab2:
        run_chat_interface()

if __name__ == "__main__":
    main()
