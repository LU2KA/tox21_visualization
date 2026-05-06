"""
TOX21 combined app — Exploratory Analysis + Benchmark Results.
Run: streamlit run streamlit_tox21_dashboard.py
"""
import streamlit as st

import streamlit_tox21_eda
import streamlit_tox21_benchmark

st.set_page_config(page_title="TOX21: EDA & Benchmark", layout="wide")
st.title("TOX21: Exploratory Analysis & Benchmark Results")

tab_eda, tab_bench = st.tabs(["Exploratory Analysis", "Benchmark Results"])

with tab_eda:
    streamlit_tox21_eda.run_eda()

with tab_bench:
    streamlit_tox21_benchmark.run_benchmark()
