import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import re
import h5py  # For HDF5 storage
import cloudpickle  # For saving/loading models in binary format
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

# Your other utility functions remain unchanged (gplearn_to_math, gplearn_to_latex, etc.)

def save_model_hdf5(model, min_max_values, filename):
    """Save model and min_max_values into HDF5 format."""
    with h5py.File(filename, 'w') as hf:
        # Save min_max_values as a dataset
        hf.create_dataset('min_max_values', data=min_max_values)
        
        # Save model using cloudpickle
        model_binary = cloudpickle.dumps(model)
        hf.create_dataset('model', data=np.void(model_binary))  # Save model as binary blob

def load_model_hdf5(filename):
    """Load model and min_max_values from HDF5 format."""
    with h5py.File(filename, 'r') as hf:
        # Load min_max_values
        min_max_values = list(hf['min_max_values'])
        
        # Load the model from the binary blob
        model_binary = hf['model'][()]
        model = cloudpickle.loads(bytes(model_binary))
        
    return model, min_max_values

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

    dot_data = est_gp._program.export_graphviz()
    st.graphviz_chart(dot_data)

    predictions = est_gp.predict(x)

    predictions = inverse_normalization(predictions, y_max, y_min)
    y = inverse_normalization(y, y_max, y_min)

    train_test_model(y, predictions)

    st.write("Faça o download do modelo:")

    # Save model and min_max_values into HDF5
    save_model_hdf5(est_gp, min_max_values, 'new_model.h5')

    # Create a download button in Streamlit
    with open('new_model.h5', 'rb') as file:
        st.download_button(
            label="Download do modelo",
            data=file,
            file_name='new_model.h5',
            mime='application/octet-stream'
        )

def predict_with_best_model(file_name, model_file=None):
    fosforo = pd.read_excel(file_name)

    if model_file is None:
        # If model_file is None, provide a default behavior (like re-training)
        st.write("No model file provided.")
        return
    else:
        # If model_file is an UploadedFile object or file path, load the model from HDF5
        model, min_max_values = load_model_hdf5(model_file)

    st.write('Teste1')
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

    dot_data = model._program.export_graphviz()
    st.graphviz_chart(dot_data)
    
    y_pred = inverse_normalization(y_pred, y_max, y_min)

    st.write("Predições:")
    st.write(y_pred)

    train_test_model(y, y_pred)
