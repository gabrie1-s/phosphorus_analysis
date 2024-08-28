import streamlit as st
import pandas as pd

import numpy as np
import seaborn as sns
import io
import sys
import re
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

def gplearn_to_latex(formula, custom_names=None):
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
            return f"\\log{{{args}}}"
        elif func == "inv":
            return f"\\frac{{1}}{{{args}}}"
        else:
            return match.group(0)

    # Regular expression to find function calls with their arguments
    pattern = re.compile(r'(\w+)\(([^()]+)\)')
    iteration = 0
    while re.search(pattern, formula):
        st.write(f"Iteration {iteration}: {formula}")
        new_formula = re.sub(pattern, replace_functions, formula)
        if new_formula == formula:  # Check if the formula is unchanged
            st.write("No further changes made; exiting loop.")
            break
        formula = new_formula
        iteration += 1

    # Replacing variable names with LaTeX friendly variables
    formula = re.sub(r'X(\d+)', r'X_\1', formula)
    
    if custom_names is not None:
        for i, custom_name in enumerate(custom_names, start=1):  # Start at 1 because X1 maps to the first variable
            formula = formula.replace(f'X_{i}', custom_name)

    return formula
    
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
  x = (x - x.min())/(x.max() - x.min())

  y = fosforo['P']
  y_min = y.min(); y_max = y.max()
  y = (y - y.min())/(y.max() - y.min())

  power = make_function(function=_power, name='power', arity=2)
  exp = make_function(function=_exp, name='exp', arity=1)
  r2 = make_fitness(function=_r2, greater_is_better=True, wrap=False)

  est_gp = SymbolicRegressor(population_size=10000,
                               generations=3, stopping_criteria=0.99,
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

  st.latex(gplearn_to_latex(str(est_gp._program), custom_names=x.columns))

  dot_data = est_gp._program.export_graphviz()
  st.graphviz_chart(dot_data)

  predictions = est_gp.predict(x)

  predictions = inverse_normalization(predictions, y_max, y_min)
  y = inverse_normalization(y, y_max, y_min)

  train_test_model(y, predictions)


st.title("Modelos para estudo de concentração do fósforo em açudes")
uploaded_file = st.file_uploader("Faça o upload da sua base de dados")
if uploaded_file:
    execute_sr(uploaded_file)
