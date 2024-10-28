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
from utils.database_utils import fetch_data_from_db 

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
    select
      "LATITUDE",
      "LONGITUDE"
    from
      sf_consumption_logs
    where
      UPPER("VESSEL_NAME") = '{vessel_name.upper()}'
      and "LATITUDE" is not null
      and "LONGITUDE" is not null
    order by
      "REPORT_DATE" desc
    limit
      1;
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
        tiles='cartodb positron',  # Light theme map
        scrollWheelZoom=True,
        dragging=True
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
        # Add custom CSS to remove extra spacing
        st.markdown(
            """
            <style>
                /* Remove extra spacing from map container */
                div[data-testid="column"] > div.element-container {
                    margin-bottom: 0 !important;
                }
                
                /* Remove spacing from metric container */
                div.stMetric {
                    margin-bottom: 0 !important;
                    padding-bottom: 0 !important;
                }
                
                /* Remove spacing from map iframe */
                iframe {
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                }
                
                /* Target the specific folium element */
                div.stFolium {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                
                /* Remove any extra margins from elements inside map container */
                div[data-testid="stExpander"] div.element-container {
                    margin: 0 !important;
                }
                
                /* Target the main element container */
                div.element-container {
                    margin-bottom: 0 !important;
                }
                
                /* Remove padding from streamlit elements */
                .css-1544g2n {
                    padding: 0 !important;
                }
                
                /* Ensure no extra padding on wrapper elements */
                .css-1r6slb0 {
                    padding: 0 !important;
                    margin: 0 !important;
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        # Create columns for position display with minimal spacing
        col1, col2 = st.columns(2)
        
        # Show coordinates
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Create and display map with specific configuration
        vessel_map = create_vessel_map(latitude, longitude)
        
        # Use st_folium with minimal configuration
        st_folium(
            vessel_map, 
            height=300,  # Reduced height
            width="100%",
            returned_objects=[],
            key="vessel_map",
        )
        
        # Add a small empty space after map (if needed)
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
    else:
        st.warning("No position data available for this vessel")

# Update the create_vessel_map function to ensure clean rendering
def create_vessel_map(latitude: float, longitude: float) -> folium.Map:
    """
    Create a Folium map centered on the vessel's position with minimal styling.
    """
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=4,
        tiles='cartodb positron',
        scrollWheelZoom=True,
        dragging=True,
        # Add minimal margins
        control_scale=True
    )
    
    # Add vessel marker
    folium.Marker(
        [latitude, longitude],
        popup='Vessel Position',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Set a fixed figure size to avoid extra spacing
    m.get_root().html.add_child(folium.Element("""
        <style>
            .folium-map {
                margin: 0 !important;
                padding: 0 !important;
            }
        </style>
    """))
    
    return m
       
def show_vessel_synopsis(vessel_name: str):
    """
    Display a comprehensive vessel synopsis including KPI summary, performance metrics,
    charts, and other relevant information.
    """
    # Create header with vessel name
    st.header(f"Vessel Synopsis - {vessel_name}")
    
    try:
        # Get hull performance data first
        hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        
        # Get speed consumption data
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        
        # Fetch CII Rating
        cii_query = f"""
        select
          cr."cii_rating"
        from
          "CII ratings" cr
          join "vessel_particulars" vp on cr."vessel_imo" = vp."vessel_imo"::bigint
        where
          vp."vessel_name" = '{vessel_name.upper()}';
        """
        cii_data = fetch_data_from_db(cii_query)
        cii_rating = cii_data.iloc[0]['cii_rating'] if not cii_data.empty else "N/A"
        
        # Fetch Vessel Score and component scores
        score_query = f"""
        select
          "Vessel Score",
          "Cost",
          "Digitalization",
          "Environment",
          "Operation",
          "Reliability"
        from
          "Vessel Scorecard"
        where
          upper("Vessels") = '{vessel_name.upper()}';
        """
        score_data = fetch_data_from_db(score_query)
        if not score_data.empty:
            vessel_score = float(score_data.iloc[0]['Vessel Score'])
            cost_score = float(score_data.iloc[0]['Cost'])
            digitalization_score = float(score_data.iloc[0]['Digitalization'])
            environment_score = float(score_data.iloc[0]['Environment'])
            operation_score = float(score_data.iloc[0]['Operation'])
            reliability_score = float(score_data.iloc[0]['Reliability'])
        else:
            vessel_score = cost_score = digitalization_score = environment_score = operation_score = reliability_score = 0.0
        
        # Fetch Crew Scores
        crew_query = """
        select 
            "Crew Skill Index",
            "Capability Index",
            "Competency Index",
            "Collaboration Index",
            "Character Index"
        from
            "crew scorecard"
        order by
            random()
        limit
            1;
        """
        crew_data = fetch_data_from_db(crew_query)
        if not crew_data.empty:
            crew_skill_index = float(crew_data.iloc[0]['Crew Skill Index'])
            capability_index = float(crew_data.iloc[0]['Capability Index'])
            competency_index = float(crew_data.iloc[0]['Competency Index'])
            collaboration_index = float(crew_data.iloc[0]['Collaboration Index'])
            character_index = float(crew_data.iloc[0]['Character Index'])
        else:
            crew_skill_index = capability_index = competency_index = collaboration_index = character_index = 0.0
        
        # Get KPI summary from LLM and display it
        kpi_summary = get_kpi_summary(
            vessel_name,
            hull_condition,
            cii_rating,
            vessel_score,
            cost_score,
            digitalization_score,
            environment_score,
            operation_score,
            reliability_score,
            crew_skill_index,
            capability_index,
            competency_index,
            collaboration_index,
            character_index
        )
        
        # Add the CSS using st.markdown
        st.markdown("""
            <style>
                .status-poor {
                    color: #dc3545;
                    font-weight: 500;
                }
                .status-average {
                    color: #ffc107;
                    font-weight: 500;
                }
                .status-good {
                    color: #28a745;
                    font-weight: 500;
                }
                .kpi-summary {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border: 1px solid #e9ecef;
                    line-height: 1.6;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Display the summary in the styled container
        st.markdown(f'<div class="kpi-summary">{kpi_summary}</div>', unsafe_allow_html=True)
        
        #st.subheader("Key Performance Indicators Summary")
        st.markdown(kpi_summary)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create expandable sections for detailed information
        with st.expander("Vessel Information", expanded=True):
            # Basic vessel info table
            st.markdown(
                f"""
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Vessel Name</td>
                        <td>{vessel_name}</td>
                    </tr>
                    <tr>
                        <td>Hull Condition</td>
                        <td>{hull_condition if hull_condition else "N/A"}</td>
                    </tr>
                    <tr>
                        <td>CII Rating</td>
                        <td>{cii_rating}</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )
        
        # Position Information
        with st.expander("Last Reported Position", expanded=True):
            show_vessel_position(vessel_name)
        
        # Performance Metrics
        with st.expander("Performance Metrics", expanded=True):
            # Hull Performance
            st.subheader("Hull Performance")
            if hull_chart:
                st.pyplot(hull_chart)
                st.markdown(hull_analysis)
            else:
                st.warning("No hull performance data available")
            
            # Speed Consumption
            st.subheader("Speed Consumption Profile")
            if speed_charts:
                st.pyplot(speed_charts)
                st.markdown(speed_analysis)
            else:
                st.warning("No speed consumption data available")
        
        # Vessel Score Details
        with st.expander("Vessel Score Details", expanded=True):
            if vessel_score > 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Display main vessel score with color indicator
                    score_color = (
                        "good" if vessel_score >= 75 
                        else "warning" if vessel_score >= 60 
                        else "critical"
                    )
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <h4>Overall Vessel Score</h4>
                            <span class='score-indicator score-{score_color}'>
                                {vessel_score:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Component scores table
                    st.markdown(
                        f"""
                        <table class="score-table">
                            <tr>
                                <th>Component</th>
                                <th>Score</th>
                                <th>Status</th>
                            </tr>
                            <tr>
                                <td>Cost</td>
                                <td>{cost_score:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if cost_score >= 75 else "warning" if cost_score >= 60 else "critical"}'>{cost_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Digitalization</td>
                                <td>{digitalization_score:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if digitalization_score >= 75 else "warning" if digitalization_score >= 60 else "critical"}'>{digitalization_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Environment</td>
                                <td>{environment_score:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if environment_score >= 75 else "warning" if environment_score >= 60 else "critical"}'>{environment_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Operation</td>
                                <td>{operation_score:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if operation_score >= 75 else "warning" if operation_score >= 60 else "critical"}'>{operation_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Reliability</td>
                                <td>{reliability_score:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if reliability_score >= 75 else "warning" if reliability_score >= 60 else "critical"}'>{reliability_score:.1f}%</span></td>
                            </tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No vessel score data available")
        
        # Crew Score Details
        with st.expander("Crew Score Details", expanded=True):
            if crew_skill_index > 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Display main crew skill index with color indicator
                    score_color = (
                        "good" if crew_skill_index >= 80 
                        else "warning" if crew_skill_index >= 70 
                        else "critical"
                    )
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <h4>Crew Skill Index</h4>
                            <span class='score-indicator score-{score_color}'>
                                {crew_skill_index:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Component indices table
                    st.markdown(
                        f"""
                        <table class="score-table">
                            <tr>
                                <th>Component</th>
                                <th>Score</th>
                                <th>Status</th>
                            </tr>
                            <tr>
                                <td>Capability</td>
                                <td>{capability_index:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if capability_index >= 80 else "warning" if capability_index >= 70 else "critical"}'>{capability_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Competency</td>
                                <td>{competency_index:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if competency_index >= 80 else "warning" if competency_index >= 70 else "critical"}'>{competency_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Collaboration</td>
                                <td>{collaboration_index:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if collaboration_index >= 80 else "warning" if collaboration_index >= 70 else "critical"}'>{collaboration_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Character</td>
                                <td>{character_index:.1f}%</td>
                                <td><span class='score-indicator score-{"good" if character_index >= 80 else "warning" if character_index >= 70 else "critical"}'>{character_index:.1f}%</span></td>
                            </tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No crew score data available")
        
        # Commercial Performance placeholder
        # with st.expander("Commercial Performance", expanded=False):
        #     st.info("Commercial performance metrics will be integrated in future updates")
        
    except Exception as e:
        st.error(f"Error generating vessel synopsis: {str(e)}")
        st.error("Please check the vessel name and try again.")
       
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

def get_kpi_summary(vessel_name: str, hull_condition: str, cii_rating: str, 
                    vessel_score: float, cost_score: float, digitalization_score: float,
                    environment_score: float, operation_score: float, reliability_score: float,
                    crew_skill_index: float, capability_index: float, competency_index: float,
                    collaboration_index: float, character_index: float) -> str:
    """
    Get specific, actionable KPI analysis with color-coded indicators.
    """
    SUMMARY_PROMPT = """
    You are a vessel performance analyst providing specific insights about a vessel's performance metrics. 
    Create a brief, conversational summary of 3-4 sentences with specific, actionable recommendations.

    Important formatting rules:
    1. Numbers should be formatted to one decimal place without the % symbol in the span tags
    2. Use words 'poor', 'average', or 'good' for status indicators
    3. Start with "Based on the data of [vessel name]"
    4. Highlight status using following rules:
       - <span class="status-poor">poor</span> for scores below 60
       - <span class="status-average">average</span> for scores 60-75
       - <span class="status-good">good</span> for scores above 75
       
    Example correct format:
    "Based on the data of [vessel name], the vessel shows <span class="status-poor">poor</span> performance with cost score at <span class="status-poor">55.4</span>."

    Current Vessel Data:
    Vessel Name: {vessel_name}
    Hull Condition: {hull_condition}
    CII Rating: {cii_rating}
    Overall Vessel Score: {vessel_score:.1f}%
    Cost Score: {cost_score:.1f}%
    Operation Score: {operation_score:.1f}%
    Crew Skill Index: {crew_skill_index:.1f}%
    Competency Index: {competency_index:.1f}%

    Focus on most critical metrics needing attention and provide specific, time-bound recommendations.
    """
    
    try:
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT.format(
                vessel_name=vessel_name,
                hull_condition=hull_condition,
                cii_rating=cii_rating,
                vessel_score=vessel_score,
                cost_score=cost_score,
                digitalization_score=digitalization_score,
                environment_score=environment_score,
                operation_score=operation_score,
                reliability_score=reliability_score,
                crew_skill_index=crew_skill_index,
                capability_index=capability_index,
                competency_index=competency_index,
                collaboration_index=collaboration_index,
                character_index=character_index
            )},
            {"role": "user", "content": "Provide a specific summary with color-coded status indicators."}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message['content'].strip()
        
    except Exception as e:
        return f"Error generating performance summary: {str(e)}"
       
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
        return f"Here's the vessel synopsis for {vessel_name}. Let me know if you need any specific information explained."  # Fixed string formatting
    
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
    # Set page config to wide mode first
    st.set_page_config(layout="wide", page_title="VesselIQ")

    # Add custom CSS to control width and spacing
    st.markdown(
        """
        <style>
            /* Set main container width and center it */
            .block-container {
                padding: 2rem 1rem;
                max-width: 60% !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            
            /* Title and header alignment */
            .stTitle {
                width: 100%;
                margin-left: 0 !important;
                padding-left: 0 !important;
            }
            
            /* Center the subtitle/description text */
            .stMarkdown p {
                width: 100%;
                margin-left: 0 !important;
                padding-left: 0 !important;
            }
            
            /* Chat container alignment */
            .stChatFloatingInputContainer {
                max-width: 60% !important;
                margin-left: auto !important;
                margin-right: auto !important;
                left: 0 !important;
                right: 0 !important;
            }
            
            /* Chat input styling */
            .stChatInputContainer {
                max-width: 100% !important;
                padding: 0 !important;
            }
            
            /* Chat messages alignment */
            .stChatMessage {
                max-width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
            }
            
            /* Message container alignment */
            .stChatMessageContainer {
                max-width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
            }
            
            /* General element container spacing */
            .element-container {
                margin-bottom: 1rem !important;
            }
            
            /* Remove default streamlit padding */
            .css-1544g2n {
                padding: 0 !important;
            }
            
            /* Ensure all content aligns properly */
            .main > .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
            
            /* Ensure expandable sections align properly */
            .stExpander {
                width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
            
            /* Ensure metric widgets align properly */
            .stMetric {
                width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Change top bar color
    st.markdown(
        """
        <script>
            const elements = window.parent.document.querySelectorAll('.main, .viewerTopBar');
            elements.forEach((element) => {
                element.style.backgroundColor = '#132337';
            });
        </script>
        """,
        unsafe_allow_html=True
    )
    
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown("Ask me about vessel performance, speed consumption, or request a complete vessel synopsis!")
    
    # Rest of the main function remains the same...
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_synopsis' not in st.session_state:
        st.session_state.show_synopsis = False
    if 'show_hull_chart' not in st.session_state:
        st.session_state.show_hull_chart = False
    if 'show_speed_charts' not in st.session_state:
        st.session_state.show_speed_charts = False
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle synopsis and chart displays
    if st.session_state.show_synopsis and 'vessel_name' in st.session_state:
        show_vessel_synopsis(st.session_state.vessel_name)
    
    if st.session_state.show_hull_chart and 'hull_chart' in st.session_state:
        st.pyplot(st.session_state.hull_chart)
    
    if st.session_state.show_speed_charts and 'speed_charts' in st.session_state:
        st.pyplot(st.session_state.speed_charts)
    
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
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
