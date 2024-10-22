import os
import openai
import streamlit as st
import json
from datetime import datetime
from config import SYSTEM_MESSAGES

def get_api_key():
    """Retrieve OpenAI API key from environment variables or Streamlit secrets"""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY in environment variables or Streamlit secrets.")
    return api_key

def create_context(cii_data, voyage_calculations=None):
    """Create a structured context from CII and voyage data"""
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

def analyze_query(query):
    """Determine the type of analysis needed based on the query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['voyage', 'route', 'plan', 'distance']):
        return 'voyage'
    elif any(word in query_lower for word in ['analyze', 'analysis', 'explain', 'why', 'how']):
        return 'analysis'
    return 'general'

def get_llm_response(user_query, context, analysis_type='general'):
    """Generate LLM response based on user query and context"""
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

def initialize_chat_history():
    """Initialize or reset chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your CII analysis assistant. I can help you understand your vessel's "
                      "carbon intensity performance and provide recommendations for improvement. "
                      "What would you like to know about your CII data?"
        })

def save_chat_history():
    """Save chat history with timestamp and metadata"""
    if hasattr(st.session_state, 'messages'):
        history = {
            'timestamp': datetime.now().isoformat(),
            'vessel_name': st.session_state.cii_data.get('vessel_name', 'Unknown'),
            'messages': st.session_state.messages,
            'metadata': {
                'cii_rating': st.session_state.cii_data.get('cii_rating', 'Unknown'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
        }
        return json.dumps(history, indent=2)
    return None

def format_response(response_text):
    """Format the response text for better readability"""
    try:
        # Add markdown formatting to numbers and metrics
        import re
        response_text = re.sub(r'(\d+\.?\d*)', r'**\1**', response_text)
        
        # Add bullet points to lists
        lines = response_text.split('\n')
        formatted_lines = []
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                formatted_lines.append('* ' + line.strip())
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except Exception:
        return response_text
