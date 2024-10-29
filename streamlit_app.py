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



def show_vessel_position(vessel_name: str):
    """
    Display the vessel's last reported position with map and coordinates.
    """
    # Get last reported position
    latitude, longitude = get_last_position(vessel_name)
    
    if latitude is not None and longitude is not None:
        # Add CSS to control spacing
        st.markdown("""
            <style>
                /* Control metric widget spacing */
                [data-testid="stMetric"] {
                    margin: 0 !important;
                    padding: 0 !important;
                    line-height: 1 !important;
                }
                [data-testid="stMetricLabel"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                [data-testid="stMetricValue"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                
                /* Control map container spacing */
                .element-container {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                div.stFolium {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                div.stFolium > iframe {
                    height: 300px !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                }
                
                /* Control column spacing */
                [data-testid="column"] {
                    padding: 0 !important;
                    margin: 0 !important;
                }
                
                /* Control expander spacing */
                [data-testid="stExpander"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                
                /* Remove any potential wrapper padding */
                .stMarkdown {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                .block-container {
                    padding-top: 0 !important;
                    padding-bottom: 0 !important;
                    margin-bottom: 0 !important;
                }
                iframe {
                    margin-bottom: -10px !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Create columns for position display
        col1, col2 = st.columns(2)
        
        # Show coordinates with minimal spacing
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Create and display map with minimal configuration
        vessel_map = create_vessel_map(latitude, longitude)
        
        # Display map with specific settings
        st_folium(
            vessel_map,
            height=300,
            width="100%",
            returned_objects=[],
            key="vessel_map",
        )
        
    else:
        st.warning("No position data available for this vessel")

def create_vessel_map(latitude: float, longitude: float) -> folium.Map:
    """
    Create a Folium map centered on the vessel's position.
    """
    # Create base map with minimal margins
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=4,
        tiles='cartodb positron',
        scrollWheelZoom=True,
        dragging=True
    )
    
    # Add marker
    folium.Marker(
        [latitude, longitude],
        popup='Vessel Position',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add custom CSS to minimize map margins
    m.get_root().html.add_child(folium.Element("""
        <style>
            .folium-map {
                margin: 0 !important;
                padding: 0 !important;
            }
            .leaflet-container {
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
        
        # Create header with vessel name
        st.header(f"Vessel Synopsis - {vessel_name.upper()}")
        
        # Add CSS for status colors and summary styling
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
                div[data-testid="stExpander"] div.element-container {
                    margin: 0 !important;
                }
                div.element-container {
                    margin-bottom: 0 !important;
                }

                /* Control expander spacing */
                [data-testid="stExpander"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                .streamlit-expanderHeader {
                    margin: 0 !important;
                    padding: 0 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Get and display KPI summary
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
        
        # Display the summary
        st.markdown(f'<div class="kpi-summary">{kpi_summary}</div>', unsafe_allow_html=True)
        
        # Create vessel info table
        with st.expander("Vessel Information", expanded=False):
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
        
        # Display vessel position
        with st.expander("Last Reported Position", expanded=False):
            show_vessel_position(vessel_name)
        
        # Display hull performance
        with st.expander("Hull Performance", expanded=False):
            if hull_chart:
                st.pyplot(hull_chart)
                st.markdown(hull_analysis)
            else:
                st.warning("No hull performance data available")
        
        # Display speed consumption
        with st.expander("Speed Consumption Profile", expanded=False):
            if speed_charts:
                st.pyplot(speed_charts)
                st.markdown(speed_analysis)
            else:
                st.warning("No speed consumption data available")
        
        # Display vessel score details
        with st.expander("Vessel Score Details", expanded=False):
            if vessel_score > 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Overall Vessel Score", f"{vessel_score:.1f}%")
                
                with col2:
                    st.markdown(
                        f"""
                        <table>
                            <tr>
                                <th>Component</th>
                                <th>Score</th>
                            </tr>
                            <tr>
                                <td>Cost</td>
                                <td><span class='status-{"Good" if cost_score >= 65 else "average" if cost_score >= 60 else "poor"}'>{cost_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Digitalization</td>
                                <td><span class='status-{"Good" if digitalization_score >= 65 else "average" if digitalization_score >= 60 else "poor"}'>{digitalization_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Environment</td>
                                <td><span class='status-{"Good" if environment_score >= 65 else "average" if environment_score >= 60 else "poor"}'>{environment_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Operation</td>
                                <td><span class='status-{"Good" if operation_score >= 65 else "average" if operation_score >= 60 else "poor"}'>{operation_score:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Reliability</td>
                                <td><span class='status-{"Good" if reliability_score >= 65 else "average" if reliability_score >= 60 else "poor"}'>{reliability_score:.1f}%</span></td>
                            </tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No vessel score data available")
        
        # Display crew score details
        with st.expander("Crew Score Details", expanded=False):
            if crew_skill_index > 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Crew Skill Index", f"{crew_skill_index:.1f}%")
                
                with col2:
                    st.markdown(
                        f"""
                        <table>
                            <tr>
                                <th>Component</th>
                                <th>Score</th>
                            </tr>
                            <tr>
                                <td>Capability</td>
                                <td><span class='status-{"good" if capability_index >= 65 else "average" if capability_index >= 60 else "poor"}'>{capability_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Competency</td>
                                <td><span class='status-{"good" if competency_index >= 65 else "average" if competency_index >= 60 else "poor"}'>{competency_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Collaboration</td>
                                <td><span class='status-{"good" if collaboration_index >= 65 else "average" if collaboration_index >= 60 else "poor"}'>{collaboration_index:.1f}%</span></td>
                            </tr>
                            <tr>
                                <td>Character</td>
                                <td><span class='status-{"good" if character_index >= 65 else "average" if character_index >= 60 else "poor"}'>{character_index:.1f}%</span></td>
                            </tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No crew score data available")
        
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
    Get comprehensive KPI analysis with proper vessel name inclusion.
    """
    SUMMARY_PROMPT = f"""
    You are a vessel performance analyst providing insights about vessel metrics. Create a comprehensive but concise summary
    addressing all major performance areas: hull condition, vessel performance scores, and crew performance.

    Important: The vessel name is "{vessel_name}" - use this exact name in your summary.

    Rules for summary:
    1. ALWAYS start with exactly: "Based on the data of {vessel_name.upper()},"
    2. Then discuss hull condition and its implications
    3. Then discuss the vessel score and its components
    4. Finally address crew performance
    5. Keep total length to 4-5 sentences maximum
    6. Prioritize the most critical issues needing immediate attention
    7. For each major issue, provide specific, time-bound recommendation

    Formatting rules:
    1. Use the exact vessel name provided above - do not use placeholders
    2. Format hull condition as:
       - <span class="status-poor">poor</span> for poor condition
       - <span class="status-average">average</span> for average condition
       - <span class="status-good">good</span> for good condition
    3. Format numeric values as:
       - <span class="status-poor">[value]</span>% for values below 60
       - <span class="status-average">[value]</span>% for values 60-75
       - <span class="status-good">[value]</span>% for values above 75
    4. Always include % symbol after the closing span tag for metrics

    Current Data for {vessel_name}:
    Hull Performance:
    - Hull Condition: {hull_condition}
    - CII Rating: {cii_rating}

    Vessel Scores (Target >65%):
    - Overall Score: {vessel_score:.1f}%
    - Cost: {cost_score:.1f}%
    - Operation: {operation_score:.1f}%
    - Environment: {environment_score:.1f}%
    - Reliability: {reliability_score:.1f}%
    - Digitalization: {digitalization_score:.1f}%

    Crew Performance (Target >65%):
    - Overall Crew Skill: {crew_skill_index:.1f}%
    - Capability: {capability_index:.1f}%
    - Competency: {competency_index:.1f}%
    - Collaboration: {collaboration_index:.1f}%
    - Character: {character_index:.1f}%

    Example format:
    "Based on the data of {vessel_name}, the hull condition is <span class="status-poor">poor</span> requiring immediate cleaning due to 15% power loss. The vessel's overall performance score is at <span class="status-poor">55.4</span>%, primarily affected by operation score at <span class="status-poor">45.6</span>% and cost efficiency at <span class="status-average">65.5</span>%. Crew performance shows <span class="status-poor">poor</span> competency at <span class="status-poor">58.4</span>%. Recommend scheduling hull cleaning within next 15 days, implementing fuel optimization program, and conducting crew technical training by end of month."

    Provide actionable insights focusing on the most critical areas requiring immediate attention.
    """
    
    try:
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Generate a comprehensive performance summary for vessel {vessel_name} prioritizing hull condition and highlighting critical areas across all KPIs."}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        # Verify vessel name is present
        summary = response.choices[0].message['content'].strip()
        if not summary.startswith(f"Based on the data of {vessel_name}"):
            summary = f"Based on the data of {vessel_name}, " + summary
            
        return summary
        
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
