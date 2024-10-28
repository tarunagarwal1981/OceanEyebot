import streamlit as st
import psycopg2
import pandas as pd

def fetch_data_from_db(query: str) -> pd.DataFrame:
    # Fetch database credentials from the nested 'supabase' section in Streamlit secrets
    conn = psycopg2.connect(
        host=st.secrets['supabase']['host'],
        database=st.secrets['supabase']['database'],
        user=st.secrets['supabase']['user'],
        password=st.secrets['supabase']['password'],
        port=st.secrets['supabase']['port']
    )
    try:
        data = pd.read_sql(query, conn)
    finally:
        conn.close()
    return data
