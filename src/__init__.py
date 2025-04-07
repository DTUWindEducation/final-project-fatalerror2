import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(PATH):
    dict = pd.read_csv(PATH)
    return dict.to_dict(orient='list')

def plot_relativehumidity_2m(PATH):
    dict = load_data(PATH)
    col1 = "Time"
    col2 = "relativehumidity_2m"
    plt.figure()
    plt.plot(dict[col1],dict[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()