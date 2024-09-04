import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import sys
import re
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

import os
import pdb
from sklearn import metrics
import plotly.graph_objects as go
import graphviz

def gplearn_to_math(formula):
    def replace_functions(match):
        func = match.group(1)
        args = match.group(2)

        if func == "add":
            return f"({args.split(',')[0]}) + ({args.split(',')[1]})"
        elif func == "sub":
            return f"({args.split(',')[0]}) - ({args.split(',')[1]})"
        elif func == "mul":
            return f"({args.split(',')[0]}) * ({args.split(',')[1]})"
        elif func == "div":
            return f"({args.split(',')[0]}) / ({args.split(',')[1]})"
        elif func == "sqrt":
            return f"sqroot({args})"
        elif func == "power":
            return f"({args.split(',')[0]})^({args.split(',')[1]})"
        elif func == "log":
            return f"ln({args})"
        elif func == "inv":
            return f"1/({args})"
        elif func == "abs":
            return f"|{args}|"
        elif func == "neg":
            return f"-({args})"
        else:
            return match.group(0)

    # Regular expression to find function calls with their arguments
    pattern = re.compile(r'(\w+)\(([^()]+)\)')
    
    while re.search(pattern, formula):
        print(formula)
        formula = re.sub(pattern, replace_functions, formula)

    # Replacing variable names with standard math notation
    formula = re.sub(r'X(\d+)', r'X\1', formula)

    return formula + " = Y"


def gplearn_to_latex(formula):
    def replace_functions(match):
        func = match.group(1)
        args = match.group(2)

        if func == "add":
            return f"{args.split(',')[0]} + {args.split(',')[1]}"
        elif func == "sub":
            return f"{args.split(',')[0]} - {args.split(',')[1]}"
        elif func == "mul":
            return f"{args.split(',')[0]} \\cdot {args.split(',')[1]}"
        elif func == "div":
            return f"\\frac{{{args.split(',')[0]}}}{{{args.split(',')[1]}}}"
        elif func == "sqrt":
            return f"\\sqrt{{{args}}}"
        elif func == "power":
            return f"{{{args.split(',')[0]}}}^{{{args.split(',')[1]}}}"
        elif func == "log":
            return f"\\log\\left({{{args}}}\\right)"
        elif func == "inv":
            return f"\\frac{{1}}{{{args}}}"
        elif func == "abs":
            return f"\\left|{args}\\right|"
        elif func == "neg":
            return f"-{{{args}}}"
        elif func == "exp":
            return f"e^{{{args}}}"
        else:
            return match.group(0)

    # Regular expression to find function calls with their arguments
    pattern = re.compile(r'(\w+)\(([^()]+)\)')
    
    while re.search(pattern, formula):
        formula = re.sub(pattern, replace_functions, formula)
        st.write(formula)

    formula = re.sub(r'X(\d+)', lambda m: f'X_{{{int(m.group(1)) + 1}}}', formula)

    return formula + ' = Y'


def show_normalization_formulas(x_min, x_max, y_min, y_max):
    st.write("Fazemos uma normalização MinMax, usando os valores dos dados de treinamento")
    for i in range(len(x_min)):
        min_val = x_min[i]
        max_val = x_max[i]
        var_name = f"B_{i+1}"
        latex_formula = rf"X_{i+1} = \frac{{{var_name} - {min_val}}}{{{max_val} - {min_val}}}"
        st.latex(latex_formula)

    min_val = y_min
    max_val = y_max
    var_name = f"P"
    latex_formula = rf"Y = \frac{{{var_name} - {min_val}}}{{{max_val} - {min_val}}}"
    st.latex(latex_formula)

def plot_results(y_pred, y_tes):
  plt.rcParams["figure.figsize"] = (10,5)
  plt.scatter(range(len(y_pred)), y_pred, c='r')
  plt.plot(range(len(y_tes)), y_tes, linestyle="-", marker="o", label="Expenses")
  plt.title('Model performance - test set')
  plt.ylabel('P medido')
  plt.xlabel('Sample')
  plt.legend(['predicted', 'real'], loc='upper left')
  fig = plt.gcf()
  st.pyplot(fig)

def train_test_model(y_te, y_pred):
    st.write("MAE: " + str(mean_absolute_error(y_te, y_pred)))
    st.write("MAPE: " + str(mean_absolute_percentage_error(y_te, y_pred)))
    st.write("MSE: " + str(mean_squared_error(y_te, y_pred)))
    st.write("R2: " + str(r2_score(y_te, y_pred)))
    plot_results(y_pred, y_te)

def inverse_normalization(v, v_max, v_min):
    return (v_max - v_min)*v + v_min
def _power(x1, x2):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        result = np.power(x1, x2)
        result = np.where(np.isfinite(result), result, 0)
        return result

def _exp(x):
    with np.errstate(over='ignore'):
        return np.where(np.isfinite(np.exp(x)), np.exp(x), 0.0)

def _r2(y, y_pred, sample_weight):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        ss_res = np.sum(sample_weight * (y - y_pred) ** 2)
        y_mean = np.average(y, weights=sample_weight)
        ss_tot = np.sum(sample_weight * (y - y_mean) ** 2)

        if np.abs(ss_tot) > 0.001:
            r2_score = 1 - ss_res / ss_tot
        else:
            r2_score = -1000

        return r2_score
    
def execute_sr(file_name):
    fosforo = pd.read_excel(file_name)

    if 'ID' in fosforo:
        fosforo.drop(columns=['ID'], inplace=True)
    fosforo.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'P']

    x = fosforo[fosforo.columns[:7]]
    y = fosforo['P']

    min_max_values = [x.min(), x.max(), y.min(), y.max()]
    
    x = (x - x.min())/(x.max() - x.min())
    y_min = y.min(); y_max = y.max()
    y = (y - y.min())/(y.max() - y.min())

    power = make_function(function=_power, name='power', arity=2)
    exp = make_function(function=_exp, name='exp', arity=1)
    r2 = make_fitness(function=_r2, greater_is_better=True, wrap=False)

    est_gp = SymbolicRegressor(population_size=10000,
                                generations=100, stopping_criteria=0.99,
                                p_crossover=0.6, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=1.0, verbose=1,
                                parsimony_coefficient=0.001, random_state=0,
                                function_set=['add', 'sub', 'mul', 'div', 'log', 'inv', 'abs', 'neg', 'sqrt', power],
                                metric=r2, n_jobs=1)

    output = io.StringIO()
    sys.stdout = output

    with st.spinner("Treinando o modelo, por favor aguarde..."):
        est_gp.fit(np.array(x), np.array(y))

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Display the captured output in Streamlit
    st.text_area("Etapas de treinamento", output.getvalue(), height=400)

    show_normalization_formulas(min_max_values[0], min_max_values[1], 
                                min_max_values[2], min_max_values[3])

    st.latex(gplearn_to_latex(str(est_gp._program)))
    # st.write(gplearn_to_math(str(est_gp._program)))

    dot_data = est_gp._program.export_graphviz()
    st.graphviz_chart(dot_data)

    predictions = est_gp.predict(x)

    predictions = inverse_normalization(predictions, y_max, y_min)
    y = inverse_normalization(y, y_max, y_min)

    train_test_model(y, predictions)

    st.write("Faça o download do modelo:")

    with open('new_model.pkl', 'wb') as file:
        pickle.dump((est_gp, min_max_values), file)

    # Create a download button in Streamlit
    with open('new_model.pkl', 'rb') as file:
        st.download_button(
            label="Download do modelo",
            data=file,
            file_name='new_model.pkl',
            mime='application/octet-stream'
        )


def predict_with_best_model(file_name, model_file=None):
    fosforo = pd.read_excel(file_name)

    if model_file is None:
        # Load the default model and min_max_values from files
        # with open('est_gp_model_1.pkl', 'rb') as f:
        #     model = pickle.load(f)

        train_dataset = pd.read_excel("Dados_B1_B7.xlsx")
        tdx = train_dataset[train_dataset.columns[:7]]
        tdy = train_dataset[train_dataset.columns[-1]]

        min_max_values = [tdx.min(), tdx.max(), tdy.min(), tdy.max()]
        tdx = (tdx - tdx.min())/(tdx.max() - tdx.min())
        tdy = (tdy - tdy.min())/(tdy.max() - tdy.min())


        phm = 0.06715839141819369; ppm = 0.04753718444355231 ;psm = 0.03765402747517174
        pc = 3.1606815034143736; ts = 0.12092886880198785

        p_cross = 1 - (psm + phm + ppm)

        power = make_function(function=_power, name='power', arity=2)
        exp = make_function(function=_exp, name='exp', arity=1)
        r2 = make_fitness(function=_r2, greater_is_better=True, wrap=False)

        with st.spinner("Recuperando o modelo, por favor aguarde..."):

            model = SymbolicRegressor(population_size=2000,
                                        tournament_size=int(ts*2000),
                                        generations=200, stopping_criteria=0.99,
                                        p_crossover=p_cross, p_subtree_mutation=psm,
                                        p_hoist_mutation=phm, p_point_mutation=ppm,
                                        verbose=1, parsimony_coefficient=0.001*pc, 
                                        random_state=0, function_set=['add','sub','mul','div','log','inv','neg','sqrt',power, exp],
                                        metric=r2, n_jobs=1)

            model.fit(np.array(tdx), np.array(tdy))
            del train_dataset, tdx, tdy


        # with open('min_max_values.pkl', 'rb') as f:
        #     min_max_values = pickle.load(f)
        #     min_max_values = list(min_max_values.values())

    else:
        if isinstance(model_file, SymbolicRegressor):
            # model_file is already a model object, so use it directly
            model = model_file
            with open('min_max_values.pkl', 'rb') as f:
                min_max_values = pickle.load(f)
        elif isinstance(model_file, st.runtime.uploaded_file_manager.UploadedFile):
            # If model_file is an UploadedFile object, load it using pickle
            model_file.seek(0)
            model, min_max_values = pickle.load(model_file)
        else:
            # If model_file is a file path, load the model and min_max_values
            with open(model_file, 'rb') as f:
                model, min_max_values = pickle.load(f)
            
    x_min, x_max, y_min, y_max = min_max_values
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    if 'ID' in fosforo:
        fosforo.drop(columns=['ID'], inplace=True)

    fosforo.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'P']
    x = fosforo[fosforo.columns[:7]]
    y = fosforo['P']

    x = (x - x_min)/(x_max - x_min)
    y_pred = model.predict(x)

    #show_normalization_formulas(x_min, x_max, y_min, y_max)
    # st.latex(gplearn_to_latex(str(model._program)))
    dot_data = model._program.export_graphviz()
    st.graphviz_chart(dot_data)
    
    
    y_pred = inverse_normalization(y_pred, y_max, y_min)

    st.write("Predições:")
    st.write(y_pred)

    train_test_model(y, y_pred)

