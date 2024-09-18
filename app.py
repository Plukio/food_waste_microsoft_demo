import streamlit as st

pg = st.navigation([st.Page('pages/Dashboard.py'), st.Page("pages/Detector.py")])
pg.run()