import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict, Tuple, Optional
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.nlp_utils import clean_vessel_name
import folium
from streamlit_folium import st_folium

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Combined performance (both hull and speed)
   - Vessel synopsis (complete vessel overview)
   - General vessel information

Choose the decision based on these rules:
- If the user asks for "vessel synopsis", "vessel summary", or "vessel overview", return "vessel_synopsis"
- If the user asks for "vessel performance" or a combination of "hull and speed performance," return "combined_performance"
- If the user asks only about "hull performance" or "hull and propeller performance," return "hull_performance"
- If the user asks only about "speed consumption," return "speed_consumption"

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "vessel_synopsis" or "general_info",
    "response_type": "concise" or "detailed",
    "explanation": "Brief explanation of why you made this decision"
}

Example responses:

Q: Show me the vessel synopsis for Nordic Aurora
{
    "vessel_name": "Nordic Aurora",
    "decision": "vessel_synopsis",
    "response_type": "detailed",
    "explanation": "User requested a complete vessel overview/synopsis"
}

Q: What's the hull performance of Oceanica Explorer?
{
    "vessel_name": "Oceanica Explorer",
    "decision": "hull_performance",
    "response_type": "concise",
    "explanation": "The query specifically asks about hull performance"
}
"""

# Function to get the OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Initialize OpenAI API
openai.api_key = get_api_key()
def get_last_position(vessel_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch the last reported position for a vessel from sf_consumption_logs.
    
    Returns:
        Tuple[Optional[float], Optional[float]]: (latitude, longitude) or (None, None) if no data
    """
    query = f"""
    SELECT TOP 1 LATITUDE, LONGITUDE
    FROM sf_consumption_logs
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    AND LATITUDE IS NOT NULL 
    AND LONGITUDE IS NOT NULL
    ORDER BY reportdate DESC
    """
    
    try:
        position_data = fetch_data_from_db(query)
        if not position_data.empty:
            return (
                float(position_data.iloc[0]['LATITUDE']),
                float(position_data.iloc[0]['LONGITUDE'])
            )
        return None, None
    except Exception as e:
        st.error(f"Error fetching position data: {str(e)}")
        return None, None

def create_vessel_map(latitude: float, longitude: float) -> folium.Map:
    """
    Create a Folium map centered on the vessel's position.
    """
    # Create a map centered on the vessel's position
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=4,
        tiles='cartodb positron'  # Light theme map
    )
    
    # Add vessel marker
    folium.Marker(
        [latitude, longitude],
        popup='Vessel Position',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

def show_vessel_position(vessel_name: str):
    """
    Display the vessel's last reported position with map and coordinates.
    """
    # Get last reported position
    latitude, longitude = get_last_position(vessel_name)
    
    if latitude is not None and longitude is not None:
        # Create columns for position display
        col1, col2 = st.columns(2)
        
        # Show coordinates
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Create and display map
        vessel_map = create_vessel_map(latitude, longitude)
        st_folium(vessel_map, height=400, width=700)
    else:
        st.warning("No position data available for this vessel")
       
def show_vessel_synopsis(vessel_name: str):
    """
    Display a comprehensive vessel synopsis including performance metrics,
    charts, and other relevant information.
    """
    # Get hull performance data
    hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
    
    # Get speed consumption data
    speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
    
    # Create header with vessel name
    st.header(f"Vessel Synopsis - {vessel_name}")
    
    # Create vessel info table
    st.subheader("Vessel Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parameter**")
        st.markdown("Vessel Name")
        st.markdown("Hull Condition")
        st.markdown("CII Rating")
    
    with col2:
        st.markdown("**Value**")
        st.markdown(vessel_name)
        st.markdown(hull_condition if hull_condition else "N/A")
        st.markdown("N/A")  # Placeholder for CII Rating
    
    # Last reported position
    st.subheader("Last Reported Position")
    show_vessel_position(vessel_name)
    
    # Hull Performance Section
    st.subheader("Hull Performance")
    if hull_chart:
        st.pyplot(hull_chart)
    else:
        st.warning("No hull performance data available")
    
    # Speed Consumption Profile
    st.subheader("Speed Consumption Profile")
    if speed_charts:
        st.pyplot(speed_charts)
    else:
        st.warning("No speed consumption data available")
    
    # Vessel Score (Placeholder)
    st.subheader("Vessel Score")
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric(label="Technical Score", value="85%")
    with score_col2:
        st.metric(label="Operational Score", value="78%")
    with score_col3:
        st.metric(label="Overall Score", value="82%")
    
    # Crew Score (Placeholder)
    st.subheader("Crew Score")
    crew_col1, crew_col2 = st.columns(2)
    with crew_col1:
        st.metric(label="Reporting Quality", value="92%")
    with crew_col2:
        st.metric(label="Response Time", value="88%")
    
    # Commercial Performance (Placeholder)
    st.subheader("Commercial Performance")
    st.info("Commercial performance metrics will be integrated in future updates")

def get_llm_decision(query: str) -> Dict[str, str]:
    """
    Get decision from LLM about query type and vessel name.
    """
    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        {"role": "user", "content": query}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
        
        decision_text = response.choices[0].message['content'].strip()
        decision_data = json.loads(decision_text)
        
        # Fallback if vessel name extraction fails
        if not decision_data.get('vessel_name'):
            match = re.search(r'of\s+(.+)', query, re.IGNORECASE)
            if match:
                decision_data['vessel_name'] = match.group(1).strip()
            else:
                decision_data['vessel_name'] = query
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        # Fallback to basic extraction
        vessel_name = re.search(r'of\s+(.+)', query, re.IGNORECASE)
        return {
            "vessel_name": vessel_name.group(1) if vessel_name else query,
            "decision": "general_info",
            "response_type": "concise",
            "explanation": "Error occurred, defaulting to general info"
        }

def handle_user_query(query: str):
    """
    Process user query and return appropriate response.
    """
    # Get decision from LLM
    decision_data = get_llm_decision(query)
    vessel_name = decision_data.get("vessel_name", "")
    decision_type = decision_data.get("decision", "general_info")
    response_type = decision_data.get("response_type", "concise")
    
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."
    
    # Store context in session state
    st.session_state.vessel_name = vessel_name
    st.session_state.decision_type = decision_type
    st.session_state.response_type = response_type
    
    # Handle different types of requests
    if decision_type == "vessel_synopsis":
        show_vessel_synopsis(vessel_name)
        return None  # No need for additional response as synopsis shows everything
    
    elif decision_type == "hull_performance":
        hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        if response_type == "concise":
            return f"The hull of {vessel_name} is in {hull_condition} condition with {power_loss:.1f}% power loss. Would you like to see detailed analysis and charts?"
        else:
            st.pyplot(hull_chart)
            return hull_analysis
    
    elif decision_type == "speed_consumption":
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        if response_type == "concise":
            return f"I've analyzed the speed consumption profile for {vessel_name}. Would you like to see the detailed analysis and charts?"
        else:
            st.pyplot(speed_charts)
            return speed_analysis
    
    elif decision_type == "combined_performance":
        hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        
        if response_type == "concise":
            return f"I have analyzed both hull and speed performance for {vessel_name}. Would you like to see the detailed analysis and charts?"
        else:
            st.pyplot(hull_chart)
            st.pyplot(speed_charts)
            return f"{hull_analysis}\n\n{speed_analysis}"
    
    else:
        return "I understand you're asking about vessel information, but could you please specify what aspect you're interested in? (hull performance, speed consumption, or complete vessel synopsis)"

def handle_follow_up(query: str):
    """
    Handle follow-up requests for more information or charts.
    """
    if 'vessel_name' not in st.session_state or 'decision_type' not in st.session_state:
        return "Could you please provide your initial question again?"
    
    vessel_name = st.session_state.vessel_name
    decision_type = st.session_state.decision_type
    
    if decision_type == "hull_performance":
        _, _, _, hull_chart = analyze_hull_performance(vessel_name)
        st.pyplot(hull_chart)
    elif decision_type == "speed_consumption":
        _, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(speed_charts)
    elif decision_type == "combined_performance":
        _, _, _, hull_chart = analyze_hull_performance(vessel_name)
        _, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(hull_chart)
        st.pyplot(speed_charts)
    elif decision_type == "vessel_synopsis":
        show_vessel_synopsis(vessel_name)

def main():
    """
    Main function for the Streamlit app.
    """
    st.title("Advanced Vessel Performance Chatbot")
    st.markdown("Ask me about vessel performance, speed consumption, or request a complete vessel synopsis!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Check if it's a follow-up request
        if re.search(r"(more|details|charts|yes)", prompt.lower()):
            handle_follow_up(prompt)
            response = "I've updated the charts and information above. Is there anything specific you'd like me to explain?"
        else:
            response = handle_user_query(prompt)
        
        if response:  # Only append response if it's not None (synopsis handles its own display)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
