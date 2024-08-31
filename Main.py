import streamlit as st
import GP_Analisys
#import poly_regression

st.title("Modelos para estudo de concentração do fósforo em açudes")
uploaded_file = st.file_uploader("Faça o upload da sua base de dados")

options = ["Selecione uma opção",
           "Treinar um novo modelo de Regressão Simbólica", 
           "Fazer predições com um modelo já treinado de Regressão Simbólica",
           #"Fazer predições com um modelo já treinado de Regressão Polinomial"
            ]

if uploaded_file:
    operation = st.selectbox("Escolha uma operação", options)

    if operation == options[1]:
        GP_Analisys.execute_sr(uploaded_file)
    elif operation == options[2]:
        options_2 = ['', 'Sim', 'Não']
        y_n = st.selectbox("Você deseja utilizar um modelo próprio?", options_2)
        
        if y_n == options_2[1]:
            model_file = st.file_uploader("Faça o upload de um arquivo .pkl", type=["pkl"])
            if model_file:
                GP_Analisys.predict_with_best_model(uploaded_file, model_file) 
                
        elif y_n == options_2[2]:
            GP_Analisys.predict_with_best_model(uploaded_file)

    # elif operation == options[3]:
    #     poly_regression.predict_with_best_model(uploaded_file)
