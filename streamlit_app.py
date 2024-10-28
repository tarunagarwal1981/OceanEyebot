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
        # Create columns for position display
        col1, col2 = st.columns(2)
        
        # Show coordinates
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Create and display map with specific configuration to prevent reruns
        vessel_map = create_vessel_map(latitude, longitude)
        st_folium(
            vessel_map, 
            height=400, 
            width="100%",
            returned_objects=[],  # This prevents reruns on map interaction
            key="vessel_map"  # Unique key to maintain map state
        )
        
        # Remove extra spacing
        st.markdown(
            """
            <style>
                .element-container .stFolium {
                    margin-bottom: -2rem;
                }
                .element-container {
                    margin-bottom: 0rem;  /* Adjusted spacing to remove extra space */
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.warning("No position data available for this vessel")
       
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
        
        # Fetch Vessel Score and other metrics
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
            competency_index = float(crew_data.iloc[0]['Competency Index'])
            capability_index = float(crew_data.iloc[0]['Capability Index'])
            collaboration_index = float(crew_data.iloc[0]['Collaboration Index'])
            character_index = float(crew_data.iloc[0]['Character Index'])
        else:
            crew_skill_index = competency_index = capability_index = collaboration_index = character_index = 0.0
        
        # Get KPI summary from LLM and display it
        kpi_summary = get_kpi_summary(
            vessel_name,
            hull_condition,
            cii_rating,
            vessel_score,
            cost_score,
            crew_skill_index,
            competency_index
        )
        
        # Display KPI Summary in a card
        st.markdown("""
            <style>
            .kpi-summary {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #e9ecef;
            }
            .kpi-summary h3 {
                color: #1a237e;
                margin-bottom: 15px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='kpi-summary'>", unsafe_allow_html=True)
        st.subheader("Key Performance Indicators Summary")
        st.markdown(kpi_summary)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create vessel info table
        st.subheader("Vessel Information")
        st.markdown(
            f"""
            <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #F4F4F4;
                padding: 8px;
                text-align: left;
            }}
            </style>
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
        
        # Last reported position
        st.subheader("Last Reported Position")
        show_vessel_position(vessel_name)
        
        # Hull Performance Section
        st.subheader("Hull Performance")
        if hull_chart:
            st.pyplot(hull_chart)
            st.markdown(hull_analysis)
        else:
            st.warning("No hull performance data available")
        
        # Speed Consumption Profile
        st.subheader("Speed Consumption Profile")
        if speed_charts:
            st.pyplot(speed_charts)
            st.markdown(speed_analysis)
        else:
            st.warning("No speed consumption data available")
        
        # Vessel Score Section
        st.subheader("Vessel Score")
        if vessel_score > 0:
            # Display main vessel score
            st.metric(label="Overall Vessel Score", value=f"{vessel_score:.1f}%")
            
            # Create a table with detailed scores
            st.markdown(
                f"""
                <style>
                .score-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 1rem;
                }}
                .score-table th, .score-table td {{
                    border: 1px solid #F4F4F4;
                    padding: 8px;
                    text-align: center;
                }}
                .score-table th {{
                    background-color: #f8f9fa;
                }}
                </style>
                <table class="score-table">
                    <tr>
                        <th>Cost</th>
                        <th>Digitalization</th>
                        <th>Environment</th>
                        <th>Operation</th>
                        <th>Reliability</th>
                    </tr>
                    <tr>
                        <td>{cost_score:.1f}%</td>
                        <td>{digitalization_score:.1f}%</td>
                        <td>{environment_score:.1f}%</td>
                        <td>{operation_score:.1f}%</td>
                        <td>{reliability_score:.1f}%</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No vessel score data available")
        
        # Crew Score Section
        st.subheader("Crew Score")
        if crew_skill_index > 0:
            # Display Crew Skill Index as metric
            st.metric(label="Crew Skill Index", value=f"{crew_skill_index:.1f}%")
            
            # Create a table with detailed indices
            st.markdown(
                f"""
                <table class="score-table">
                    <tr>
                        <th>Index Type</th>
                        <th>Score</th>
                    </tr>
                    <tr>
                        <td>Capability Index</td>
                        <td>{capability_index:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Competency Index</td>
                        <td>{competency_index:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Collaboration Index</td>
                        <td>{collaboration_index:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Character Index</td>
                        <td>{character_index:.1f}%</td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No crew score data available")
        
        # Commercial Performance Section
        st.subheader("Commercial Performance")
        st.info("Commercial performance metrics will be integrated in future updates")
        
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
                    vessel_score: float, cost_score: float, crew_skill_index: float,
                    competency_index: float) -> str:
    """
    Get KPI analysis and recommendations from LLM.
    """
    SUMMARY_PROMPT = """
    You are a vessel performance analyst. Based on the following KPIs for the vessel, provide a brief bullet-point summary
    with one-line status and recommendation for each KPI. Focus on actionable insights.

    Vessel KPIs:
    - Hull Condition: {hull_condition}
    - CII Rating: {cii_rating}
    - Vessel Score: {vessel_score}%
    - Cost Score: {cost_score}%
    - Crew Skill Index: {crew_skill_index}%
    - Crew Competency Index: {competency_index}%

    Format your response as bullet points, with each point containing Status: and Recommendation:
    Keep each bullet point concise (1-2 lines max).
    Focus only on notable issues that require attention or excellent performance worth maintaining.
    """
    
    try:
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT.format(
                hull_condition=hull_condition,
                cii_rating=cii_rating,
                vessel_score=vessel_score,
                cost_score=cost_score,
                crew_skill_index=crew_skill_index,
                competency_index=competency_index
            )},
            {"role": "user", "content": "Generate a KPI summary with recommendations."}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message['content'].strip()
        
    except Exception as e:
        return f"Error generating KPI summary: {str(e)}"

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

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
            .element-container {
                margin-bottom: 1rem;
            }
            .stMarkdown {
                margin-bottom: 0rem;
            }
            .stMetric {
                margin-bottom: 0.5rem;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # JavaScript to change the top bar background color
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
    
    # Initialize session state variables if they don't exist
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
    
    # Show synopsis if requested
    if st.session_state.show_synopsis and 'vessel_name' in st.session_state:
        show_vessel_synopsis(st.session_state.vessel_name)
    
    # Show charts if requested
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
        
        if response:  # Only append response if it's not None
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# Add these functions to clear state when needed
def clear_charts():
    """Clear chart-related session state"""
    st.session_state.show_hull_chart = False
    st.session_state.show_speed_charts = False
    if 'hull_chart' in st.session_state:
        del st.session_state.hull_chart
    if 'speed_charts' in st.session_state:
        del st.session_state.speed_charts

def clear_synopsis():
    """Clear synopsis-related session state"""
    st.session_state.show_synopsis = False
   
if __name__ == "__main__":
    main()
