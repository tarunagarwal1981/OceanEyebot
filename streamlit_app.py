import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.nlp_utils import clean_vessel_name

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Combined performance (both hull and speed)
   - General vessel information

Additionally:
- If the user asks for "vessel performance" or a combination of "hull and speed performance," return "combined_performance."
- If the user asks only about "hull performance" or "hull and propeller performance," return "hull_performance."
- If the user asks only about "speed consumption," return "speed_consumption."

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "general_info",
    "explanation": "Brief explanation of why you made this decision"
}


# Example logic for how to handle the response:

Example 1:
Q: Can you give me the hull performance of Oceanica Explorer?
{
    "vessel_name": "Oceanica Explorer",
    "decision": "hull_performance",
    "response_type": "concise",
    "explanation": "The query asks about hull performance and seems to expect a concise response."
}

Example 2:
Q: Show me the detailed performance and charts for Starlight Voyager.
{
    "vessel_name": "Starlight Voyager",
    "decision": "combined_performance",
    "response_type": "detailed",
    "explanation": "The query specifically asks for detailed performance and charts, so a more comprehensive response is needed."
}

Example 2:
Q: Show me the vessel performance and charts for Starlight Voyager.
{
    "vessel_name": "Starlight Voyager",
    "decision": "combined_performance",
    "response_type": "detailed",
    "explanation": "The query specifically asks for detailed performance and charts, so a more comprehensive response is needed."
}
"""


# LLM Prompts and Few-Shot Examples
FEW_SHOT_EXAMPLES = """
Example 1:
Q: What's the hull condition of the vessel Oceanica Explorer?
A: The hull of Oceanica Explorer is in **average condition**, with an excess power requirement of **8.7%**, indicating moderate fouling. The next cleaning is scheduled for **2023-11-15**. 
   Would you like a detailed analysis and charts for this vessel?

Follow-up (detailed request):
Q: Yes, give me more details.
A: Here's a detailed analysis of the hull performance for Oceanica Explorer:
1. **Current Excess Power**: 8.7% more power is needed to maintain the vessel's speed compared to a clean hull condition.
2. **Fouling Rate**: Power loss is increasing at 0.5% per month, suggesting moderate fouling.
3. **Hull Condition**: Overall, the hull condition is rated as **Average**.
4. **Forecasted Hull Cleaning**: Scheduled for 2023-11-15.
5. **Performance Impact**: The excess power results in approximately 6-7% increased fuel consumption, assuming normal operations.
   
**Charts**: Below are the performance charts for further analysis.
*Remember: Accurate reporting is crucial for precise analysis.*

Example 2:
Q: Can you give me the hull and propeller performance of Starlight Voyager?
A: The hull and propeller of Starlight Voyager are in **good condition**. Current excess power is only **2.5%**, indicating minimal fouling.
   Would you like to see a detailed report and performance charts?

Follow-up (detailed request):
Q: Yes, give me the detailed report.
A: Here's the detailed analysis of hull and propeller performance for Starlight Voyager:
1. **Hull Condition**: Good, with minimal fouling.
2. **Propeller Condition**: No significant performance impact observed.
3. **Excess Power**: 2.5% above clean hull condition.
4. **Performance Impact**: The vessel is operating efficiently, with a slight increase in fuel consumption.

**Charts**: Below are the charts for detailed performance insights.
*Accurate reporting ensures the vessel continues to perform optimally.*

Example 3:
Q: What is the hull performance of Sea Breeze?
A: The hull performance of Sea Breeze is **average**, with around **6% excess power** required to maintain speed. The vessel's fouling rate is consistent, and a hull cleaning is recommended within the next month. 
   Would you like a detailed analysis and charts?

Follow-up (detailed request):
Q: Yes, give me charts and more details.
A: Here's the detailed hull performance analysis for Sea Breeze:
1. **Current Excess Power**: 6% more power is required due to fouling.
2. **Fouling Rate**: The fouling rate is consistent, indicating steady power loss over time.
3. **Hull Condition**: The hull is rated as **Average**, with cleaning recommended within the next month.
4. **Impact on Fuel Consumption**: The excess power is leading to approximately 5-6% more fuel consumption.

**Charts**: Below are the performance charts for further reference.
*Make sure to remind the crew to report data accurately for better analysis.*

Example 4:
Q: Can you provide the speed consumption profile for the vessel Starlight Voyager?
A: I've analyzed the speed consumption data for Starlight Voyager. Here's what I found:

1. Speed Range: The vessel operates between 10 to 18 knots based on the available data.
2. Consumption Trend: There's a clear non-linear increase in fuel consumption as speed increases.
3. Optimal Speed: The most fuel-efficient speed appears to be around 12-13 knots, where the increase in consumption per knot is lowest.
4. High-Speed Impact: Operating at speeds above 16 knots results in a sharp increase in fuel consumption, potentially over 50% more than at optimal speed.
5. Loading Conditions: The data shows distinct consumption profiles for laden and ballast conditions, with ballast condition generally showing lower consumption at the same speeds.

Recommendations:
1. Prioritize operating at speeds between 12-13 knots when possible to maximize fuel efficiency.
2. For time-sensitive operations, consider the trade-off between increased speed and fuel consumption, especially above 16 knots.
3. Optimize route planning to take advantage of the more efficient ballast condition where applicable.
4. Monitor and record speed and consumption data regularly to identify any deviations from this profile, which could indicate performance issues.
5. Consider conducting a detailed analysis of the economic impact of speed on your specific trade routes to find the optimal balance between speed and efficiency.

**Additionally, here's the speed consumption chart for your reference.**

*Please note: This analysis is as good as the data reported by the vessel. We kindly request you to remind the vessel to report data accurately and periodically. Accurate analysis not only helps reduce the vessel's carbon footprint but also realizes fuel cost savings.*

Guidelines for Analysis:
- If the user asks about vessel performance or hull performance, always include **both the speed consumption chart and hull performance chart** for reference.
- If the hull condition is gauged as **Good**, do not recommend any hull cleaning or underwater inspection.
- If the hull condition is **Average**, recommend performing an underwater hull inspection and propeller polishing.
- If the hull condition is **Poor**, recommend performing hull cleaning and propeller polishing.
- After each response, include a polite reminder: *This analysis is as good as the data reported by the vessel. We kindly request you to remind the vessel to report data accurately and periodically. Accurate analysis not only helps reduce the vessel's carbon footprint but also realizes fuel cost savings.*

Now, please answer the following question in a similar style, using the data I provide:
{user_question}

{data_summary}

Provide a detailed analysis and recommendations based on this data.
"""

# LLM detailed analysis function
def get_llm_analysis(query: str, hull_analysis: str, speed_analysis: str, hull_condition: str) -> str:
    # Prepare the vessel data summary based on hull and speed analysis
    data_summary = f"Hull Analysis: {hull_analysis}\nSpeed Analysis: {speed_analysis}\nHull Condition: {hull_condition}"

    # LLM prompt including few-shot examples
    prompt = f"""
    {FEW_SHOT_EXAMPLES}

    User Question: {query}

    Vessel Data Summary:
    {data_summary}

    Provide a detailed analysis and recommendations based on this data.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a vessel performance analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        st.error(f"Error in LLM analysis: {str(e)}")
        return "An error occurred during the analysis."




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

# Fallback regex to extract vessel name in case LLM fails
def fallback_extract_vessel_name(query: str) -> str:
    match = re.search(r'of\s+(.+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return query  # If regex fails, return the original query

# Function to call ChatGPT for decision making using ChatCompletion
def get_llm_decision(query: str) -> Dict[str, str]:
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
        #st.write(f"LLM Response: {decision_text}")  # Debugging output
        
        decision_data = json.loads(decision_text)
        
        # Fallback if the vessel name is not correctly extracted
        if "vessel_name" not in decision_data or decision_data['vessel_name'] is None or 'hull performance' in decision_data['vessel_name'].lower():
            decision_data['vessel_name'] = fallback_extract_vessel_name(query)
        
        return decision_data
    except openai.error.InvalidRequestError as e:
        st.error(f"InvalidRequestError: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "Invalid request. Defaulting to general info."
        }
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "An error occurred. Defaulting to general info."
        }

# Function to handle user query and return analysis
def handle_user_query(query: str):
    # Get the decision and vessel name from the LLM (ChatGPT)
    llm_decision = get_llm_decision(query)

    # Safeguard: Ensure the response contains vessel_name and decision
    vessel_name = llm_decision.get("vessel_name", "")
    answer_type = llm_decision.get("answer_type", "concise")
    decision_type = llm_decision.get("decision", "general_info")

    # Check if the vessel name was extracted correctly
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."

    # Store the vessel name and decision type in session state for follow-up queries
    st.session_state.vessel_name = vessel_name
    st.session_state.decision_type = decision_type

    # Now based on the answer_type (concise/detailed), respond accordingly
    if answer_type == "concise":
        if decision_type == 'hull_performance':
            hull_analysis, power_loss_pct, hull_condition, _ = analyze_hull_performance(vessel_name)
            analysis = f"The hull of {vessel_name} is in {hull_condition} condition with {power_loss_pct:.2f}% power loss. Would you like more details or charts?"

        elif decision_type == 'speed_consumption':
            speed_analysis, _ = analyze_speed_consumption(vessel_name)
            analysis = f"The speed consumption of {vessel_name} shows a clear trend. Would you like more details or charts?"

        elif decision_type == 'combined_performance':
            hull_analysis, _, hull_condition, _ = analyze_hull_performance(vessel_name)
            speed_analysis, _ = analyze_speed_consumption(vessel_name)
            analysis = f"Both hull and speed data for {vessel_name} indicate good performance. Would you like a detailed report or charts?"

        else:
            analysis = "The query seems to require general vessel information or is unclear. Please refine the query."

    elif answer_type == "detailed":
        # Handle detailed requests
        handle_more_information()  # Provide detailed analysis and charts if requested
    
    return analysis


# Function to handle follow-up queries asking for more information
def handle_more_information():
    # Check if vessel name and decision type are in session state
    if 'vessel_name' in st.session_state and 'decision_type' in st.session_state:
        vessel_name = st.session_state.vessel_name
        decision_type = st.session_state.decision_type

        # Based on the stored decision type, provide the detailed response
        if decision_type == 'hull_performance':
            hull_analysis, power_loss_pct, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
            detailed_analysis = get_llm_analysis(f"Hull performance of {vessel_name}", hull_analysis, "", hull_condition)
            st.pyplot(hull_chart)

        elif decision_type == 'speed_consumption':
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            detailed_analysis = get_llm_analysis(f"Speed consumption of {vessel_name}", "", speed_analysis, "")
            st.pyplot(speed_chart)

        elif decision_type == 'combined_performance':
            hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            detailed_analysis = get_llm_analysis(f"Combined performance of {vessel_name}", hull_analysis, speed_analysis, hull_condition)
            st.pyplot(hull_chart)
            st.pyplot(speed_chart)
        
        st.write(detailed_analysis)
    else:
        st.warning("No previous context found. Please provide a new query.")

# Function to display the charts based on the LLM's decision
def display_charts(decision: str, vessel_name: str):
    if decision in ["speed_consumption", "combined_performance"]:
        try:
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            if speed_chart is not None and hasattr(speed_chart, 'savefig'):
                st.pyplot(speed_chart)
            else:
                st.warning("Speed consumption chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the speed consumption chart: {str(e)}")
    
    if decision in ["hull_performance", "combined_performance"]:
        try:
            # Unpack the returned values from the hull performance agent
            hull_analysis, _, _, hull_chart = analyze_hull_performance(vessel_name)
            if hull_chart is not None and hasattr(hull_chart, 'savefig'):
                st.pyplot(hull_chart)
            else:
                st.warning("Hull performance chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the hull performance chart: {str(e)}")


# Main function for the Streamlit app
def main():
    st.title("Advanced Vessel Performance Chatbot (Powered by ChatGPT)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        # Check if it's a follow-up request like "show charts" or "give me more information"
        if re.search(r"(more information|give me charts|detailed|yes)", prompt, re.IGNORECASE):
            handle_more_information()
        else:
            # Handle initial query
            analysis = handle_user_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": analysis})
            with st.chat_message("assistant"):
                st.markdown(analysis)

# Run the app
if __name__ == "__main__":
    main()
