import streamlit as st
import import_ipynb
import GP_Analisys

st.title("Modelos para estudo de concentração do fósforo em açudes")
uploaded_file = st.file_uploader("Faça o upload da sua base de dados")
if uploaded_file:
    GP_Analisys.execute_sr(uploaded_file)